import argparse
from copy import deepcopy

from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

from QuantAttention import CustomizGPT2Attention, CustomizedQwen2Attention
from utils.GPT2_pruner_quantizer import GPT2_PRUNER

import torch
import torch.nn as nn
from torch.optim import AdamW, RMSprop, SGD
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchmetrics.classification import MulticlassAccuracy

from torch.cuda.amp import GradScaler, autocast

import wandb

from tqdm.auto import tqdm

import config as cfg

import bitsandbytes as bnb

#Huggingface
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification
from transformers import DataCollatorForLanguageModeling
from transformers import get_scheduler, EvalPrediction
from transformers import default_data_collator, DataCollatorWithPadding
from transformers.trainer_pt_utils import get_parameter_names

from transformers import TrainingArguments



from datasets import load_dataset
from accelerate import Accelerator

import sys
import numpy as np  

# import torchviz
# import graphviz

task_to_keys = {
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "qqp":  ("question1", "question2"),
    "sst2": ("sentence", None),
}



def print_environment_info():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
    else:
        print("CUDA is not available")

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

def str_int_and_none(value):
    try:
        # Try to convert the value to an integer
        return int(value)
    except ValueError:
        # If conversion to int fails, return the value as a string
        if value.casefold() == "none".casefold():
            return None
        else:
            return value
    
def float_and_none(value):
    try:
        # Try to convert the value to a float
        return float(value)
    except ValueError:
        # If conversion to float fails, return the value as a string
        if value.casefold() == "none".casefold():
            return None
        else:
            raise ValueError(f"Unsupported value type {value}")

def main(args):

    # Load the dataset
    raw_datasets = load_dataset(
        "glue",
        args.task_name,
        trust_remote_code=True,
    )

    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)

    # load the model and tokenizer

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    try:
        if args.model_name == "gpt2":       
            tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", trust_remote_code=True, use_fast=True)
            config = AutoConfig.from_pretrained(
                "openai-community/gpt2",
                num_labels=num_labels,
                finetuning_task=args.task_name,
                # cache_dir=args.cache_dir,
                trust_remote_code=True,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                "openai-community/gpt2", 
                config=config,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )

        elif args.model_name == "Qwen_1.5b":
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B", trust_remote_code=True, use_fast=True)
            config = AutoConfig.from_pretrained(
                "Qwen/Qwen2-1.5B",
                num_labels=num_labels,
                finetuning_task=args.task_name,
                # cache_dir=args.cache_dir,
                trust_remote_code=True,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                "Qwen/Qwen2-1.5B", 
                config=config,
                attn_implementation="eager",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map='auto'
            )
        elif args.model_name == "Qwen_0.5b":
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True, use_fast=True)
            config = AutoConfig.from_pretrained(
                "Qwen/Qwen2-0.5B",
                num_labels=num_labels,
                finetuning_task=args.task_name,
                # cache_dir=args.cache_dir,
                trust_remote_code=True,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                "Qwen/Qwen2-0.5B", 
                config=config,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                # variant="fp16",
                device_map='auto'
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

    wandb.login()

    wandb.init(project='SQS_GLUE', name=args.run_name)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')


    sentence1_key, sentence2_key = task_to_keys[args.task_name]

    padding = "max_length" if args.pad_to_max_length else False

    if args.max_seq_length > tokenizer.model_max_length:
        print(
            f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

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
            num_proc=args.preprocessing_num_workers,
            remove_columns=raw_datasets["train"].column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
    )


    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    # dataLoaders creation:
    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        if args.precision == "amp_fp16" or args.precision == "amp_bf16":
            use_fp16 = True
        else:
            use_fp16 = False
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if use_fp16 else None))

    # train_sampler = DistributedSampler(train_dataset, shuffle=True)
    # eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
    
    train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.batch_size)

    num_train_epochs = args.duration
    num_update_steps_per_epoch = len(train_dataloader)
    cfg.TOT_TRAIN_STEP = num_training_steps = num_train_epochs * num_update_steps_per_epoch
    cfg.PRUNE_END_STEP = int(len(train_dataloader)*args.prune_end)
    cfg.PRUNE_START_STEP = int(len(train_dataloader)*args.prune_start)

    accelerator = Accelerator(
        mixed_precision= 'fp16' if use_fp16 else 'bf16',
        # fp16 = TrainingArguments.fp16,
        device_placement=True,
    )

    def evaluate(model, eval_dataloader):
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
    accuracy = evaluate(model, eval_dataloader)
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

    progress_bar = tqdm(range(num_training_steps))


    for epoch in range(args.duration):
        model.train()
        cfg.IS_TRAIN = True
        for step, batch in enumerate(train_dataloader):

            curr_step = len(train_dataloader)*epoch+step
            if not args.normal:
                pruner.prune(curr_step)
                pruner.log_sparsity()

            # for key in batch:
            #     print(f"{key}: {batch[key]}")
            with accelerator.autocast():
                outputs = model(**batch)
                loss = outputs.loss

            wandb.log({'Training Loss': loss.item()})

            # accelerator.scaler.scale(loss).backward()  # ✅ Use scaler for mixed precision training
            # accelerator.scaler.step(optimizer)  # ✅ Step with scaler
            # accelerator.scaler.update()  # ✅ Update scaler

            accelerator.backward(loss)

            for name, param in model.named_parameters():
                print(f"{name} grad dtype: {param.grad.dtype}")
                print(f"{name} dtype: {param.dtype}")

            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            if step % 50 == 0:
                accuracy = evaluate(model, eval_dataloader)
                wandb.log({'Eval Acc': accuracy}, commit=False)

        accelerator.wait_for_everyone()
        # if pruner.cur_sparsity == args.final_sparsity:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.save_folder+"epoch"+str(epoch), save_function=accelerator.save)



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
    parser.add_argument("--train_file", type=str, default=None, 
                        help="A csv or a json file containing the training data.")
    parser.add_argument("--validation_file", type=str, default=None, 
                        help="A csv or a json file containing the validation data.")
    parser.add_argument("--test_file", type=str, default=None, 
                        help="A csv or a json file containing the Prediction data.")
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

    main(args)