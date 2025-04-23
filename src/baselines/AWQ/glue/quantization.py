import argparse
import os
import logging
from logging import getLogger
import datetime


import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


from datasets import load_dataset

from tqdm.auto import tqdm

import SQS.config as cfg
from SQS.config import model_config


import bitsandbytes as bnb

#Huggingface
from transformers import AutoConfig
from transformers import default_data_collator, DataCollatorWithPadding
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)





task_to_keys = {
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "qqp":  ("question1", "question2"),
    "sst2": ("sentence", None),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from utils import *

def setup_logger(log_dir="./logs", log_filename=None):
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_filename}_{timestamp}.log"
    
    log_path = os.path.join(log_dir, log_filename)
    
    logger = getLogger(__name__)
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)

    # Stream handler (console)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
    


def config_glue_dataset(task_name, tokenizer, batch_size, eval_batch_size, raw_datasets, max_seq_length, pad_to_max_length,
                        preprocessing_num_workers, overwrite_cache):
    sentence1_key, sentence2_key = task_to_keys[task_name]

    padding = "max_length" if pad_to_max_length else False

    if args.max_seq_length > tokenizer.model_max_length:
        print(
            f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=max_seq_length, truncation=True)

        if "label" in examples:
            # in all cases, rename the column to labels because the model will expect that.
            result["labels"] = examples["label"]
        return result
    

    processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=raw_datasets["train"].column_names,
            load_from_cache_file=not overwrite_cache,
            desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if task_name == "mnli" else "validation"]

    g = torch.Generator().manual_seed(42)
    sample_data_indices = torch.randperm(len(train_dataset), generator=g)[:8000]
    subset_dataset = train_dataset.select(sample_data_indices.tolist())

    # dataLoaders creation:
    if pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer)

    train_dataloader = DataLoader(subset_dataset, collate_fn=data_collator, batch_size=batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=eval_batch_size)
    return train_dataloader, eval_dataloader, processed_datasets

def evaluate(model, eval_dataloader, device, num_labels):
    model.eval()
    cfg.IS_TRAIN = False
    metric = MulticlassAccuracy(num_classes=num_labels, average='micro').to(device)

    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items() if torch.is_tensor(v)}
            
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            metric.update(predictions, batch['labels'])
    
    accuracy = metric.compute()
    cfg.IS_TRAIN = True
    return accuracy

def main(args):

    
    logger = setup_logger(log_dir="{}/logs/{}/{}".format(args.base_dir, args.model_name, args.task_name), log_filename=args.method)

    #Load Model an Tokenizer

    raw_datasets = load_dataset("glue", args.task_name, trust_remote_code=True)

    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",         # <-- AWQ quantization
    #     bnb_4bit_compute_dtype=torch.float16,
    # )


    try:
        loaded_model_config= model_config[args.model_name]
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=True)
        config = AutoConfig.from_pretrained(
                args.model_name,
                num_labels=num_labels,
                finetuning_task=args.task_name,
                # cache_dir=args.cache_dir,
                trust_remote_code=True,
            )
        model = AutoModelForSequenceClassification.from_pretrained(
                loaded_model_config['from_pretrained'], 
                config=config,
                attn_implementation=loaded_model_config['attn_implementation'],
                trust_remote_code=True,
                # quantization_config=bnb_config,
            )
        
        print("Defining pad token")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    
    # Config Dataloader
    train_dataloader, eval_dataloader, processed_datasets = config_glue_dataset(args.task_name, tokenizer, args.batch_size, args.eval_batch_size, raw_datasets, args.max_seq_length, args.pad_to_max_length, args.preprocessing_num_workers, args.overwrite_cache)
    
    
    quant_path = args.save_folder+'/{}'.format(args.model_name)
    quant_config = {
        "zero_point": True,
        "per_channel": True,
        "q_group_size": 128,
        "w_bit": args.bits,  
    }

    # logger.info("="*20+"Quantizing Model"+"="*20)
    # model.quantize(quant_path, quant_config)
    # logger.info("="*20+"Quantization Done"+"="*20)

    eval_acc = evaluate(model, eval_dataloader, device, num_labels)
    logger.info(f"Eval Accuracy: {eval_acc}")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--bits", type=int, help="Number of bits to quantize to")

    parser.add_argument('--save_folder', type=str,
                        help="Output dir.")

    parser.add_argument('--preprocessing_num_workers', type=int, default=4,
                        help="Number of workers for preprocessing.")

    parser.add_argument('--task_name', type=str, default=None, help='Task Name in GLUE')
    parser.add_argument('--num_labels', type=int, default=2,
                        help='Number of labels')
    parser.add_argument('--pad_to_max_length', action="store_true",
                        help="If passed, pad all samples to `max_seq_length`. Otherwise, dynamic padding is used.")

    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument("--overwrite_cache", action="store_true", 
                        help="Overwrite the cached training and evaluation sets")
    
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help="Max sequence length.")

    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch Size')
    parser.add_argument('--eval_batch_size', type=int, default=128,
                        help='Evaluation Batch Size')
    parser.add_argument("--model_name", type=str, default="openai-community/gpt2",
                        choices=["openai-community/gpt2", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-0.5B", 'meta-llama/Llama-3.2-1B'],
                        help="Backbone Model Name.")
    parser.add_argument('--method', type=str, default="SQS",
                        choices=["AWQ", "Normal"], help='Choose Method')
    parser.add_argument('--base_dir', type=str, default="SQS-H100",
                        help='Base Directory')

 
    args = parser.parse_args()
    main(args)

