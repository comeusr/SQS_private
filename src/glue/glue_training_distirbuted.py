import argparse
from copy import deepcopy
import os
import time

import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW, RMSprop, SGD
from torch.utils.data import DataLoader, Subset
from torchmetrics.classification import MulticlassAccuracy
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


from datasets import load_dataset

from tqdm.auto import tqdm

import config as cfg
from config import model_config

import bitsandbytes as bnb

#Huggingface
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import get_scheduler
from transformers import default_data_collator, DataCollatorWithPadding
from transformers.trainer_pt_utils import get_parameter_names

from transformers import TrainingArguments

from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP

from QuantAttention import CustomizGPT2Attention, CustomizedQwen2Attention, CustomizedLlamaAttention, CustomizedLLamaMLP
from utils.GPT2_pruner_quantizer import GPT2_PRUNER


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

def setup(rank, world_size):
    # Set up environment variables for distributed training.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # Initialize the distributed process group.
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # Set the current GPU device.
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def model_memory_summary(model):
    total_params = 0
    total_memory = 0  # In bytes
    
    print(f"{'Layer':<50}{'Shape':<30}{'Memory (MB)'}")
    print("=" * 100)
    
    for name, param in model.named_parameters():
        shape = tuple(param.shape)
        num_params = param.numel()
        memory = num_params * param.element_size() / (1024 ** 2)  # Convert bytes to MB
        total_params += num_params
        total_memory += num_params * param.element_size()
        
        print(f"{name:<50}{str(shape):<30}{memory:.4f}")
    
    for name, buffer in model.named_buffers():
        shape = tuple(buffer.shape)
        num_params = buffer.numel()
        memory = num_params * buffer.element_size() / (1024 ** 2)  # Convert bytes to MB
        total_memory += num_params * buffer.element_size()
        
        print(f"{name:<50}{str(shape):<30}{memory:.4f}")
        print("=" * 100)
        print(f"Total Parameters: {total_params}")
        print(f"Total Memory Consumption: {total_memory / (1024 ** 2):.4f} MB")


def recursive_setattr(obj, attr, value):
    attr = attr.split('.', 1)
    if len(attr) == 1:
        setattr(obj, attr[0], value)
    else:
        recursive_setattr(getattr(obj, attr[0]), attr[1], value)

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
        elif isinstance(m, CustomizedLlamaAttention):
            print("Initializing Layer {}".format(count))
            print("Initializing Customized Model Parameters.")
            m.init_mask_params(sigma) 
            count += 1 
        elif isinstance(m, CustomizedLLamaMLP):
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
    elif model_name == "meta-llama/Llama-3.2-1B":
        if isinstance(module, LlamaAttention):
            target_state_dict   = deepcopy(module.state_dict())
            new_module          = CustomizedLlamaAttention(config, layer_idx=module.layer_idx).to(device)
            new_module.load_state_dict(target_state_dict)
            print("Replace with Customize Llama Flash Attention Layer.")
            return new_module
        elif isinstance(module, LlamaMLP):
            target_state_dict   = deepcopy(module.state_dict())
            new_module          = CustomizedLLamaMLP(config).to(device)
            new_module.load_state_dict(target_state_dict)
            print("Replace with Customize Llama MLP Layer.")
            return new_module
        else:
            return module


def config_glue_dataset(task_name, precision, tokenizer, batch_size, eval_batch_size, raw_datasets, max_seq_length, pad_to_max_length,
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
    sample_data_indices = torch.randperm(len(train_dataset), generator=g)[:10000]

    # dataLoaders creation:
    if pad_to_max_length:
        data_collator = default_data_collator
    else:
        if precision == "amp_fp16" or precision == "amp_bf16":
            use_fp16 = True
        else:
            use_fp16 = False
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if use_fp16 else None))

    train_dataloader = DataLoader(train_dataset[sample_data_indices], collate_fn=data_collator, batch_size=batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=eval_batch_size)
    return train_dataloader, eval_dataloader, processed_datasets

def main(args):

    if args.distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        setup(rank, world_size)
    
    # Load the dataset
    raw_datasets = load_dataset("glue", args.task_name, trust_remote_code=True)

    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)

    # load the model and tokenizer
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
                torch_dtype=torch.float32,
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

    if args.freeze_weight:
        for name, param in model.named_parameters():
            param.requires_grad = False



    train_dataloader, eval_dataloader, processed_datasets = config_glue_dataset(args.task_name, args.precision, tokenizer, args.batch_size, args.eval_batch_size, raw_datasets, args.max_seq_length, args.pad_to_max_length,
                        args.preprocessing_num_workers, args.overwrite_cache)

    num_train_epochs = args.duration
    num_update_steps_per_epoch = len(train_dataloader)
    cfg.TOT_TRAIN_STEP = num_training_steps = num_train_epochs * num_update_steps_per_epoch
    cfg.PRUNE_END_STEP = int(len(train_dataloader)*args.prune_end)
    cfg.PRUNE_START_STEP = int(len(train_dataloader)*args.prune_start)

    if args.distributed:
        model = FSDP(model, device_ids=[rank])
    else:
        model.to(device)
    
    if not args.normal:
        for name, module in tuple(model.named_modules()):
            if name:
                recursive_setattr(model, name, replace_attn_layer(module, config, args.model_name, device))
        print("Initializing Model Parameters.")
        InitModel(model, args.sigma)
    
    print(model)

    for module in model.modules():
        module.to(device)

    # for name, param in model.named_parameters():
    #     print(name, param.device)


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
                "params": [p for n, p in model.named_parameters() if "mu" in n],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if 'mu' not in n],
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
        
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if "mu" in name],
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if "mu" not in name],
                "lr": args.lr,
                "weight_decay": 0.0,
            },
        ]
        optimizer = SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    print("Model Device: ", model.device)

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

    # model_memory_summary(model)

    model_train(rank, world_size, train_dataloader, eval_dataloader, model, pruner, optimizer, lr_scheduler, 
                num_training_steps, args.duration, args.normal, cfg, device, num_labels)

def model_train(rank, world_size, train_dataloader, eval_dataloader, model, pruner, optimizer, lr_scheduler,
                num_training_steps,duration, normal, cfg, device, num_labels):
    progress_bar = tqdm(range(num_training_steps))

    print(f"Running DDP training on rank {rank}.")
    setup(rank, world_size)



    for epoch in range(duration):
        model.train()
        cfg.IS_TRAIN = True
        for step, batch in enumerate(train_dataloader):

            curr_step = len(train_dataloader)*epoch+step
            
            time_start = time.time()
            if not normal:
                pruner.prune(curr_step)
                pruner.log_sparsity()
            time_end = time.time()
            print("-"*50+"Step {} Pruning Time taken {:.4f}".format(step, time_end-time_start)+"-"*50)

            time_start = time.time()
            outputs = model(**batch)
            time_end = time.time()
            print("-"*50+"Step {} Forward Time taken {:.4f}".format(step, time_end-time_start)+"-"*50)

            loss = outputs.loss

            wandb.log({'Training Loss': loss.item()})

            optimizer.zero_grad()

            time_start = time.time()
            loss.backward(loss, retain_graph=True)
            time_end = time.time()

            print("-"*50+"Step {} Backward Time taken {:.4f}".format(step, time_end-time_start)+"-"*50)

            for name, param in model.named_parameters():
                if param.grad is not None:
                    if param.grad.device != param.device:
                        param.grad = param.grad.to(param.device)
                    # print(name+" grad device", param.grad.device)
                    # print(name+" device", param.device)


            for name, param in model.named_parameters():
                if param.requires_grad:
                    if param.grad is None:
                        print(f"{name} grad: {param.grad}")
                    elif torch.isnan(param.grad).any():
                        print("-"*50+"NaN Grad Found"+"-"*50)
                        print(f"{name} grad: {param.grad}")
            
            time_start = time.time()
            if not normal:
                pruner.apply_non_prune_gradient(curr_step)
            time_end = time.time()
            print("-"*50+"Step {} Pruner Apply Gradient Time taken {:.4f}".format(step, time_end-time_start)+"-"*50)

            time_start = time.time()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            time_end = time.time()
            print("-"*50+"Step {} Clip Grad Norm Time taken {:.4f}".format(step, time_end-time_start)+"-"*50)

            # if step  % 10 == 0:
            #     for name, param in model.named_parameters():
            #         if param.requires_grad:
            #             print(name+".grad", np.linalg.norm(param.grad.detach().cpu().numpy()))

            time_start = time.time()
            optimizer.step()
            time_end = time.time()
            print("-"*50+"Step {} Optimizer Step Time taken {:.4f}".format(step, time_end-time_start)+"-"*50)

            time_start = time.time()
            lr_scheduler.step()
            time_end = time.time()
            print("-"*50+"Step {} Lr Scheduler Step Time taken {:.4f}".format(step, time_end-time_start)+"-"*50)

            progress_bar.update(1)

            if rank == 0:
                if step % 10 == 0:
                    print("-"*50+"Evaluating Model"+"-"*50)
                    accuracy = evaluate(model, eval_dataloader, device, num_labels)
                    wandb.log({'Eval Acc': accuracy}, commit=False)

        # if pruner.cur_sparsity == args.final_sparsity:
        check_point_path = args.save_folder+"epoch"+str(epoch)+"ddp_model.pth"
        if args.distributed:
            if rank == 0:
                torch.save(model.module.state_dict(), check_point_path)
        else:
            torch.save(model.state_dict(), args.save_folder+"epoch"+str(epoch)+"model.pth")
          

def evaluate(model, eval_dataloader, device, num_labels):
    model.eval()
    cfg.IS_TRAIN = False
    metric = MulticlassAccuracy(num_classes=num_labels, average='micro').to(device)

    with torch.no_grad():
        for batch in eval_dataloader:
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
    parser.add_argument('--eval_batch_size', type=int, default=128,
                        help='Evaluation Batch Size')
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
    parser.add_argument("--model_name", type=str, default="openai-community/gpt2",
                        choices=["openai-community/gpt2", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-0.5B", 'meta-llama/Llama-3.2-1B'],
                        help="Backbone Model Name.")
    parser.add_argument('--prior', type=str, default="spike_slab",
                        choices=['spike_slab', 'normal'],
                        help='Choose Prior for the KL divergence')
    parser.add_argument('--optimizer', type=str, default="rmsprop",
                        choices=["adam", "rmsprop", "sgd"],
                        help='Choose Optimizer')
    parser.add_argument('--distributed', action='store_true', default=False,
                        help='Distributed Training or Not.')

    args = parser.parse_args()
    wandb.login()

    wandb.init(project='SQS_GLUE', name=args.run_name)
    main(args)