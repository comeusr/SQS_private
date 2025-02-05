import argparse
from copy import deepcopy



import torch
import torch.nn as nn
from torch.optim import AdamW, RMSprop, SGD
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.classification import MulticlassAccuracy
from torch.cuda.amp import GradScaler, autocast

from datasets import load_dataset
from accelerate import Accelerator

from tqdm.auto import tqdm

import config as cfg
from config import model_config

import bitsandbytes as bnb

#Huggingface
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification
from transformers import DataCollatorForLanguageModeling
from transformers import get_scheduler, EvalPrediction
from transformers import default_data_collator, DataCollatorWithPadding
from transformers.trainer_pt_utils import get_parameter_names

from transformers import TrainingArguments

from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

from QuantAttention import CustomizGPT2Attention, CustomizedQwen2Attention
from utils.GPT2_pruner_quantizer import GPT2_PRUNER



import sys
import numpy as np  

from utils import *

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

import wandb

task_to_keys = {
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "qqp":  ("question1", "question2"),
    "sst2": ("sentence", None),
}


def InitModel(model, sigma):
    count = 1
    for name, m in model.named_modules():
        if isinstance(m, CustomizGPT2Attention):
            print("Initializing Customized Model Parameters.")
            m.init_mask_params(sigma)
            count += 1
        elif isinstance(m, CustomizedQwen2Attention):
            print("Initializing Layer {}".format(count))
            print("Initializing Customized Model Parameters.")
            m.init_mask_params(sigma) 
            count += 1 

def replace_attn_layer(module, config, model_name, device):
    if model_name == "gpt2":
        if isinstance(module, GPT2Attention):
            target_state_dict   = deepcopy(module.state_dict())
            new_module          = CustomizGPT2Attention(config, is_cross_attention=module.is_cross_attention, )
            new_module.load_state_dict(target_state_dict)
            print("Replace with Customize GPT Attention Layer.")
            return new_module
        else:
            return module
    elif model_name == "Qwen_1.5b" or model_name == "Qwen_0.5b":
        if isinstance(module, Qwen2Attention):
            # target_state_dict   = deepcopy(module.state_dict())
            new_module = CustomizedQwen2Attention(config, layer_idx=module.layer_idx).to(device)
            new_module.load_state_dict(module.state_dict())
            print("Replace with Customize Qwen Flash Attention Layer.")
            return new_module
        else:
            return module


def config_glue_dataset(task_name, precision, tokenizer, batch_size, raw_datasets, max_seq_length, pad_to_max_length,
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

    # dataLoaders creation:
    if pad_to_max_length:
        data_collator = default_data_collator
    else:
        if precision == "amp_fp16" or precision == "amp_bf16":
            use_fp16 = True
        else:
            use_fp16 = False
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if use_fp16 else None))

    # train_sampler = DistributedSampler(train_dataset, shuffle=True)
    # eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
    
    train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, batch_size=batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size)
    return train_dataloader, eval_dataloader

def main(args):

    # Load the dataset
    raw_datasets = load_dataset("glue", args.task_name, trust_remote_code=True)

    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)

    # load the model and tokenizer

    try:
        loaded_model_config= model_config[args.model_name]
        tokenizer = AutoTokenizer.from_pretrained(loaded_model_config['from_pretrained'], trust_remote_code=True, use_fast=True)
        config = AutoConfig.from_pretrained(
                loaded_model_config['from_pretrained'],
                num_labels=num_labels,
                finetuning_task=args.task_name,
                # cache_dir=args.cache_dir,
                trust_remote_code=True,
            )
        model = AutoModelForSequenceClassification.from_pretrained(
                loaded_model_config['from_pretrained'], 
                config=config,
                attn_implementation=loaded_model_config['attn_implementation'],
                # torch_dtype=torch.float16,
                trust_remote_code=True
            )
        print("Defining pad token")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    print(model)
    # set the evluation metrics based on the task
    cfg.set_config(args=args)


    config_glue_dataset(args.task_name, args.precision, tokenizer, arg.sbatch_size, raw_datasets, args.max_seq_length, args.pad_to_max_length,
                        args.preprocessing_num_workers, args.overwrite_cache)

    num_train_epochs = args.duration
    num_update_steps_per_epoch = len(train_dataloader)
    cfg.TOT_TRAIN_STEP = num_training_steps = num_train_epochs * num_update_steps_per_epoch
    cfg.PRUNE_END_STEP = int(len(train_dataloader)*args.prune_end)
    cfg.PRUNE_START_STEP = int(len(train_dataloader)*args.prune_start)

    accelerator = Accelerator(
        # mixed_precision= 'fp16' if use_fp16 else 'bf16',
        # fp16 = TrainingArguments.fp16,
        device_placement=True,
    )



    if args.task_name == "mnli":
        mm_eval_dataset = processed_datasets["validation_mismatched"]
        mm_eval_sampler = DistributedSampler(mm_eval_dataset, shuffle=False)
        mm_eval_dataloader = DataLoader(mm_eval_dataset, collate_fn=data_collator, batch_size=args.batch_size, sampler=mm_eval_sampler)
        # TODO: add the customized evaluator

        # mnli_matched_task = Evaluator(
        #     label='mnli_matched_accuracy',
        #     dataloader=eval_dataloader,
        #     metric_names=['MulticlassAccuracy']
        # )
        # mnli_mismatched_task = Evaluator(
        #     label='mnli_mismatched_accuracy',
        #     dataloader=mm_eval_dataloader,
        #     metric_names=['MulticlassAccuracy']
        # )

    if args.optimizer == "adam":
        print("AdamW optimizer")
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            eps=1e-8
        )
        decay_parameters = get_parameter_names(model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if n in decay_parameters],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]

        optimizer_kwargs = {
            "eps": 1e-8,
        }
        optimizer_kwargs["lr"] = args.lr

        optimizer = bnb.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            eps=1e-8
        )
    elif args.optimizer == "rmsprop":
        print("SGD optimizer")
        optimizer = RMSprop(
            model.parameters(),
            lr=args.lr,
            eps=1e-10,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "sgd":
        print("SGD optimizer")
        optimizer = SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )


    # training_args = TrainingArguments(
    #         output_dir=args.save_folder,
    #         per_device_train_batch_size=1,
    #         gradient_accumulation_steps=4,
    #         gradient_checkpointing=True,
    #         fp16=True,
    #     )

    # model = torch.compile(model)

    if not args.normal:
        for name, module in tuple(model.named_modules()):
            if name:
                recursive_setattr(model, name, replace_attn_layer(module, config, args.model_name, device))
        print("Initializing Model Parameters.")
        InitModel(model, args.sigma)

    model.to(accelerator.device, dtype=torch.float16)

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    print("Model Device: ", model.device)
    accuracy = evaluate(model, eval_dataloader, accelerator, device, num_labels)
    wandb.log({'Before Compression Eval Acc': accuracy}, commit=False)

    pruner =  GPT2_PRUNER(model, 
                          init_sparsity=args.init_sparsity , 
                          final_sparsity=args.final_sparsity, 
                          alpha_f=0.1)
    
    lr_scheduler = get_scheduler(
        'cosine',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=cfg.PRUNE_END_STEP,
    )

    model_train(train_dataloader, eval_dataloader, model, pruner, optimizer, accelerator, lr_scheduler, 
                num_training_steps, args.duration, args.normal, cfg, device, num_labels)

def model_train(train_dataloader,eval_dataloader, model, pruner, optimizer, accelerator,lr_scheduler,
                num_training_steps,duration, normal, cfg, device, num_labels):
    progress_bar = tqdm(range(num_training_steps))


    for epoch in range(duration):
        model.train()
        cfg.IS_TRAIN = True
        for step, batch in enumerate(train_dataloader):

            curr_step = len(train_dataloader)*epoch+step
            if not normal:
                pruner.prune(curr_step)
                pruner.log_sparsity()

            # for key in batch:
            #     print(f"{key}: {batch[key]}")
            # with accelerator.autocast():
            outputs = model(**batch)
            loss = outputs.loss

            wandb.log({'Training Loss': loss.item()})

            # accelerator.scaler.scale(loss).backward()  # ✅ Use scaler for mixed precision training
            # accelerator.scaler.step(optimizer)  # ✅ Step with scaler
            # accelerator.scaler.update()  # ✅ Update scaler

            accelerator.backward(loss)

            # for name, param in model.named_parameters():
            #     print(f"{name} grad dtype: {param.grad.dtype}")
            #     print(f"{name} dtype: {param.dtype}")

            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            if step % 50 == 0:
                accuracy = evaluate(model, eval_dataloader, accelerator, device, num_labels)
                wandb.log({'Eval Acc': accuracy}, commit=False)

        accelerator.wait_for_everyone()
        # if pruner.cur_sparsity == args.final_sparsity:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.save_folder+"epoch"+str(epoch), save_function=accelerator.save)


def evaluate(model, eval_dataloader, accelerator, device, num_labels):
    model.eval()
    cfg.IS_TRAIN = False
    metric = MulticlassAccuracy(num_classes=num_labels, average='micro').to(device)

    with torch.no_grad():
        for batch in eval_dataloader:

            # for key in batch:
            #     batch[key] = batch[key].to(torch.bfloat16)
            # for key in batch:
            #     print(f"{key}: {batch[key]}")
            with accelerator.autocast():
                outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            metric.update(predictions, batch['labels'])
    
    accuracy = metric.compute()
    cfg.IS_TRAIN = True
    return accuracy

if __name__ == "__main__":
    # print_environment_info()
    parser = argparse.ArgumentParser(description="Unify Pruning and Quantization on Language Models",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=512,
                        help="Seed Number.")
    parser.add_argument('--save_folder', type=str,
                        help="Output dir.")
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help="Max sequence length.")
    parser.add_argument('--preprocessing_num_workers', type=int, default=4,
                        help="Number of workers for preprocessing.")
    parser.add_argument('--run_name', type=str, default="BERT Quantization and Pruning.",
                        help="Run name")
    parser.add_argument('--precision', type=str, default="amp_fp16",
                        help="Precision")
    parser.add_argument('--project_name', type=str,
                        help='Projct name used for wandb.')
    parser.add_argument('--watch', action='store_true', default=False,
                        help='Whether choose to watch the paremters of model.')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='Dataset')
    parser.add_argument('--task_name', type=str, default=None, help='Task Name in GLUE')
    parser.add_argument('--num_labels', type=int, default=2,
                        help='Number of labels')
    parser.add_argument('--pad_to_max_length', action="store_true",
                        help="If passed, pad all samples to `max_seq_length`. Otherwise, dynamic padding is used.")
    parser.add_argument('--autoresume', action='store_true', default=False,
                        help="Auto Resume the training process.")
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument("--overwrite_cache", action="store_true", 
                        help="Overwrite the cached training and evaluation sets")
    
    parser.add_argument('--empirical', type=bool, default=False,
                        help='whether use empirical initialization for parameter sigma (default: False)')
    parser.add_argument('--normal', action='store_true', default=False,
                        help='whether train noramlly (default: False)')
    parser.add_argument('--K', type=int, default=4, metavar='K',
                    help='number of GMM components')
    parser.add_argument('--tau', type=float, default=0.01, metavar='TAU',
                        help='gumbel softmax temperature (default: 0.01)')
    parser.add_argument('--init_method', type=str, default='k-means',
                        choices=['k-means', 'quantile', 'empirical'],
                        help='Choose Initialization Method for Mu')
    parser.add_argument('--prune', action='store_true', default=False,
                        help="Prune or Not")
    parser.add_argument('--prune_scale', type=float, default=0.2,
                        help='Scale the pruning parameter by 1/prune_scale')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='When debug skip initialization')
    parser.add_argument('--freeze_weight', action='store_true', default=False,
                        help='Freeze Parameters')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch Size')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help="Initial Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--duration', type=int, default=20,
                        help="Number of Epochs")
    parser.add_argument('--warm_up', type=str, default='2ep',
                        help='Warm Up epoch before pruning')
    parser.add_argument('--init_sparsity', type=float, default=0.0,
                        help='Begin with this intial step size.')
    parser.add_argument('--final_sparsity', type=float, default=0.7,
                        help='The target sparsity.')
    parser.add_argument('--pr_warm_up', type=str, default='2ep',
                        help='Warm Up epoch before pruning')
    parser.add_argument('--sigma', type=float, default=3,
                        help="Initial Sigma for the prior distribution.")
    parser.add_argument('--doc_stride', type=int, default=128,
                        help='stride length for features')
    parser.add_argument('--alpha_f', type=float, default=0.01,
                        help='The ratio between the final LR and initial LR.')
    parser.add_argument('--eval_interval', type=str, default='5ep',
                        help='Frequence of validation.')
    parser.add_argument('--prune_end', type=float, default=3,
                        help='End of the Pruning')
    parser.add_argument('--prune_start', type=float, default=1, 
                        help='Starting point of the pruning process.')
    parser.add_argument('--sample', action='store_true', default=False,
                        help = "Use Bayesian Sample or Not")
    parser.add_argument('--pretrain_path', type=str, default=None,
                        help="Path to load pretrained model.")
    parser.add_argument('--average', action='store_true', default=False,
                        help="Whether use Bayesian Average to ensemble model.")
    parser.add_argument("--model_name", type=str, default="gpt2",
                        choices=["gpt2", "Qwen_1.5b", "Qwen_0.5b"],
                        help="Backbone Model Name.")
    parser.add_argument('--prior', type=str, default="spike_slab",
                        choices=['spike_slab', 'normal'],
                        help='Choose Prior for the KL divergence')
    parser.add_argument('--optimizer', type=str, default="rmsprop",
                        choices=["adam", "rmsprop", "sgd"],
                        help='Choose Optimizer')

    args = parser.parse_args()
    wandb.login()

    wandb.init(project='SQS_GLUE', name=args.run_name)
    main(args)