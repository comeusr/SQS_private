CUDA_LAUNCH_BLOCKING=1

import argparse
from copy import deepcopy

from transformers.models.gpt2.modeling_gpt2 import GPT2SdpaAttention
from transformers.models.qwen2.modeling_qwen2 import Qwen2FlashAttention2

from QuantAttention import CustomizGPT2SdpaAttention, CustomizedQwenFlashAttention2
from utils.GPT2_pruner_quantizer import GPT2_PRUNER

import torch
import torch.nn as nn
from torch.optim import AdamW, RMSprop, SGD
from torch.utils.data import DataLoader

import wandb

from tqdm.auto import tqdm

import config as cfg

#Huggingface
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import get_scheduler, EvalPrediction


from datasets import load_dataset
from accelerate import Accelerator

import sys
import numpy as np  

import torchviz
import graphviz

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

def replace_attn_layer(module, config, model_name, device):
    if model_name == "gpt2":
        if isinstance(module, GPT2SdpaAttention):
            target_state_dict   = deepcopy(module.state_dict())
            new_module          = CustomizGPT2SdpaAttention(config, is_cross_attention=module.is_cross_attention, )
            new_module.load_state_dict(target_state_dict)
            print("Replace with Customize GPT Attention Layer.")
            return new_module
        else:
            return module
    elif model_name == "Qwen_1.5b" or model_name == "Qwen_0.5b":
        if isinstance(module, Qwen2FlashAttention2):
            # target_state_dict   = deepcopy(module.state_dict())
            new_module = CustomizedQwenFlashAttention2(config, layer_idx=module.layer_idx).to(device)
            new_module.load_state_dict(module.state_dict())
            print("Replace with Customize Qwen Flash Attention Layer.")
            return new_module
        else:
            return module

    
def recursive_setattr(obj, attr, value):
    attr = attr.split('.', 1)
    if len(attr) == 1:
        setattr(obj, attr[0], value)
    else:
        recursive_setattr(getattr(obj, attr[0]), attr[1], value)

def InitModel(model:nn.Module, sigma):
    count = 1
    for name, m in model.named_modules():
        if isinstance(m, CustomizGPT2SdpaAttention):
            print("Initializing Customized Model Parameters.")
            m.init_mask_params(sigma)
            count += 1
        elif isinstance(m, CustomizedQwenFlashAttention2):
            print("Initializing Layer {}".format(count))
            print("Initializing Customized Model Parameters.")
            m.init_mask_params(sigma) 
            count += 1 


def main():
    print_environment_info()
    parser = argparse.ArgumentParser(description="Unify Pruning and Quantization on Language Models",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=512,
                        help="Seed Number.")
    parser.add_argument('--save_folder', type=str,
                        help="Output dir.")
    parser.add_argument('--run_name', type=str, default="BERT Quantization and Pruning.",
                        help="Run name")
    parser.add_argument('--project_name', type=str,
                        help='Projct name used for wandb.')
    parser.add_argument('--watch', action='store_true', default=False,
                        help='Whether choose to watch the paremters of model.')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='Dataset')
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
    parser.add_argument('--max_length', type=int, default=512,
                        help="max length of LLM model features")
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
    
    cfg.set_config(args=args)

    wandb.login()

    wandb.init(project='SQS_LLM', name=args.run_name)


    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    try:
        if args.model_name == "gpt2":       
            tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                "openai-community/gpt2", 
                attn_implementation="flash_attention_2",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        elif args.model_name == "Qwen_1.5b":
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2-1.5B", 
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map='auto'
            )
        elif args.model_name == "Qwen_0.5b":
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2-0.5B", 
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map='auto'
            )
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    print(model)

    for name, module in model.named_modules():
        if isinstance(module, Qwen2FlashAttention2):
            wandb.log({name+"_query": wandb.Histogram(module.q_proj.weight.data.cpu().numpy(), num_bins=128)}, commit=False)
            wandb.log({name+"_key": wandb.Histogram(module.k_proj.weight.data.cpu().numpy(), num_bins=128)}, commit=False)
            wandb.log({name+"_value": wandb.Histogram(module.v_proj.weight.data.cpu().numpy(), num_bins=128)}, commit=False)
            wandb.log({name+"_output": wandb.Histogram(module.o_proj.weight.data.cpu().numpy(), num_bins=128)}, commit=False)

            print("Histogram of {}".format(name))
            histogram = np.histogram(module.q_proj.weight.data.cpu().numpy(), bins=args.K)
            print(histogram)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = min(args.max_length, tokenizer.model_max_length)
    doc_stride = args.doc_stride

    config = model.config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if not args.normal:
        for name, module in tuple(model.named_modules()):
            if name:
                recursive_setattr(model, name, replace_attn_layer(module, config, args.model_name, device))

    if not args.pretrain_path:
        InitModel(model, args.sigma)
    else:
        model.from_pretrained(args.pretrain_path)

    
    for name, module in model.named_modules():
        if isinstance(module, CustomizedQwenFlashAttention2):
            print('-'*20+"{}".format(name)+"-"*20)
            print(module.q_proj.sub_distribution.mu.data.cpu().numpy())

    if args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.

            ###################################
            #  Use cached dataset if possible #
            ###################################
        if args.dataset_name == "wikitext-103-v1":
            raw_dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", cache_dir="/scratch/gilbreth/wang4538/cache", trust_remote_code=True)
        elif args.dataset_name == "ptb_text_only":
            raw_dataset = load_dataset("ptb_text_only", cache_dir="/scratch/gilbreth/wang4538/cache", trust_remote_code=True)
        else:
            raw_dataset = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir="/scratch/gilbreth/wang4538/cache")
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
            extension = args.validation_file.split(".")[-1]
        if args.test_file is not None:
            data_files["test"] = args.test_file
            extension = args.test_file.split(".")[-1]
        raw_dataset = load_dataset(extension, data_files=data_files, field="data")

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


    def tokenize_function(example):
        if args.dataset_name == "ptb_text_only":
            example = example['sentence']
        elif args.dataset_name == "wikitext-103-v1":
            example = example['text']
        
        return tokenizer(example, 
                         max_length=max_length,
                         return_special_tokens_mask=False, 
                         padding="max_length",
                         add_special_tokens=True,
                         truncation=True,
                         return_tensors='pt',
                         return_overflowing_tokens=False
                        )
    
    tokenized_train_data = raw_dataset['train'].map(tokenize_function,
                                          batched=True,
                                          remove_columns=raw_dataset['train'].column_names)

    tokenized_validation_data = raw_dataset['validation'].map(tokenize_function, 
                                                             batched=True, 
                                                             remove_columns=raw_dataset['validation'].column_names
                                                             )
    
    tokenized_train_data.set_format('torch')
    tokenized_validation_data.set_format('torch')

    train_dataloader = DataLoader(
        tokenized_train_data,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.batch_size,
        pin_memory=True,
    )

    eval_dataloader = DataLoader(
        tokenized_validation_data,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=8,
        pin_memory=True,
    )


    num_train_epochs = args.duration
    num_update_steps_per_epoch = len(train_dataloader)
    cfg.TOT_TRAIN_STEP = num_training_steps = num_train_epochs * num_update_steps_per_epoch
    cfg.PRUNE_END_STEP = int(len(train_dataloader)*args.prune_end)
    cfg.PRUNE_START_STEP = int(len(train_dataloader)*args.prune_start)


    if args.optimizer == "adam":
        print("AdamW optimizer")
        optimizer = AdamW(
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

    accelerator = Accelerator(
        mixed_precision='bf16',
        device_placement=True,
    )
    # model = torch.compile(model)
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # for param in model.parameters():
    #     param.required_grad = False

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


    def evaluate(model, eval_dataloader):
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)


            losses.append(accelerator.gather(outputs.loss).reshape(1)) 

        loss = torch.mean(torch.cat(losses))
        try:
            perplexity = torch.exp(loss)
            print("line 384: perplexity = ", perplexity)
        except OverflowError:
            perplexity = float("inf")
            print("in except, error!!!!!!!!!!!!!!!!")
        return loss.item(), perplexity.item()
    
    progress_bar = tqdm(range(num_training_steps))

    # Evaluate before Train
    # model.eval()
    # cfg.IS_TRAIN = False
    # eval_loss, eval_ppl = evaluate(model, eval_dataloader)
    # wandb.log({'Validation Loss': eval_loss}, commit=False)
    # wandb.log({'Validation PPL': eval_ppl}, commit=False)

    for epoch in range(args.duration):

        # if args.watch:
        #     watch_quantize_weight(model)
        model.train()
        wandb.log({'epoch': epoch}, commit=False)
        cfg.IS_TRAIN=True


        for step, batch in enumerate(train_dataloader):

            for name, module in model.named_modules():
                if isinstance(module, CustomizedQwenFlashAttention2):
                    wandb.log({name+"_query_mu": wandb.Histogram(module.q_proj.sub_distribution.mu.data.cpu().numpy(), num_bins=args.K)}, commit=False)
                    wandb.log({name+"_key_mu": wandb.Histogram(module.k_proj.sub_distribution.mu.data.cpu().numpy(), num_bins=args.K)}, commit=False)
                    wandb.log({name+"_value_mu": wandb.Histogram(module.v_proj.sub_distribution.mu.data.cpu().numpy(), num_bins=args.K)}, commit=False)
                    wandb.log({name+"_output_mu": wandb.Histogram(module.o_proj.sub_distribution.mu.data.cpu().numpy(), num_bins=args.K)}, commit=False)

                    SoftKey, SoftValue, SoftQuery, SoftOutput = module.get_Sweight()
                    wandb.log({name+"_key_soft": wandb.Histogram(SoftKey.data.cpu().numpy())}, commit=False)
                    wandb.log({name+"_value_soft": wandb.Histogram(SoftValue.data.cpu().numpy())}, commit=False)
                    wandb.log({name+"_query_soft": wandb.Histogram(SoftQuery.data.cpu().numpy())}, commit=False)
                    wandb.log({name+"_output_soft": wandb.Histogram(SoftOutput.data.cpu().numpy())}, commit=False)


            curr_step = len(train_dataloader)*epoch+step
            pruner.prune(curr_step)
            pruner.log_sparsity()
            # pruner.monitor_scheduler_step(optimizer)

            outputs = model(**batch)

            loss = outputs.loss
            # graph = torchviz.make_dot(loss, params=dict(model.named_parameters()))
            # graph.render("graph", format="pdf")
            # # graph.view()
            # print("Torchviz Render Done")
            
            # print("Graphviz Rendering")
            # dot = graphviz.Source(graph.source)
            # dot.render("graph1", format="pdf")
            # print("Graphviz Rende Done")


            wandb.log({'Training Loss': loss.item()})
            # accelerator.backward(loss)

            # # for name, param in model.named_parameters():
            # #     if param.grad is not None:
            # #         if param.grad.isnan().any():
            # #             print("-"*40+"{} param grad is nan".format(name)+"-"*40)
            # #             # print(param.grad)
            # #         else:
            # #             print("-"*40+"{} param grad is not nan".format(name)+"-"*40)
            # #     else:
            # #         print("-"*40+"{} param grad is None".format(name)+"-"*40)
            # pruner.apply_non_prune_gradient(step)
            # accelerator.clip_grad_norm_(model.parameters(), 1.0)
            # for name, param in model.named_parameters():
            #     if "embed_tokens" in name:
            #         param.requires_grad = False

            #     print(name+" requires_grad: ", param.requires_grad)
            #     try:
            #         if param.grad.isnan().any():
            #             print("-"*40+"{} param grad is nan".format(name)+"-"*40)
            #             print(param.grad)
            #     except:
            #         print(name)


            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            
        model.eval()
        cfg.IS_TRAIN = False
        # Evaluate Model on validation dataset. 
        eval_loss, eval_ppl = evaluate(model, eval_dataloader)

        wandb.log({'Validation Loss': eval_loss}, commit=False)
        wandb.log({'Validation PPL': eval_ppl}, commit=False)

        accelerator.wait_for_everyone()
        if pruner.cur_sparsity == args.final_sparsity:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.save_folder+"_epoch", save_function=accelerator.save)


if __name__ == "__main__":
    main()

