import argparse
from copy import deepcopy

from transformers.models.gpt2.modeling_gpt2 import GPT2SdpaAttention

from QuantAttention import CustomizGPT2SdpaAttention
from utils.GPT2_pruner_quantizer import GPT2_PRUNER

import torch
import torch.nn as nn
from torch.optim import AdamW
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

def replace_attn_layer(module, config):
    if isinstance(module, GPT2SdpaAttention):
        target_state_dict   = deepcopy(module.state_dict())
        new_module          = CustomizGPT2SdpaAttention(config, is_cross_attention=module.is_cross_attention, )
        new_module.load_state_dict(target_state_dict)
        return new_module
    else:
        return module
    
def recursive_setattr(obj, attr, value):
    attr = attr.split('.', 1)
    if len(attr) == 1:
        setattr(obj, attr[0], value)
    else:
        recursive_setattr(getattr(obj, attr[0]), attr[1], value)

def InitGPT2Model(model:nn.Module, sigma):
    for name, m in model.named_modules():
        if isinstance(m, CustomizGPT2SdpaAttention):
            m.init_mask_params(sigma)


# def watch_quantize_weight(model):
    
#     for name, m in model.named_modules():
#         if isinstance(m, CustomizGPT2SdpaAttention):
#             query_mu, key_mu,value_mu = m.getMu()
#             querySweight, keySweight, valueSweight = m.get_Sweight()
#             queryPweight, keyPweight, valuePweight = m.get_Pweight()

#             wandb.log({name+"_S_query": wandb.Histogram(querySweight.data.cpu().numpy())}, commit=False)
#             wandb.log({name+"_S_key": wandb.Histogram(keySweight.data.cpu().numpy())}, commit=False)
#             wandb.log({name+"_S_value": wandb.Histogram(valueSweight.data.cpu().numpy())}, commit=False)


#             wandb.log({name+"_Mu_query": wandb.Histogram(query_mu)}, commit=False)
#             wandb.log({name+"_Mu_key": wandb.Histogram(key_mu)}, commit=False)
#             wandb.log({name+"_Mu_value": wandb.Histogram(value_mu)}, commit=False)



def main():
    parser = argparse.ArgumentParser(description="Unify Pruning and Quantization via VI on BERT model",
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
                    help='number of GMM components (default: 2^4=16)')
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
    parser.add_argument('--max_length', type=int, default=384,
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

    args = parser.parse_args()
    
    cfg.set_config(args=args)

    wandb.login()

    wandb.init(project='Quantization_GPT2', name=args.run_name)


    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = min(args.max_length, tokenizer.model_max_length)
    doc_stride = args.doc_stride

    config = model.config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)


    for name, module in tuple(model.named_modules()):
        if name:
            recursive_setattr(model, name, replace_attn_layer(module, config))

    InitGPT2Model(model, args.sigma)

    if args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.

            ###################################
            #  Use cached dataset if possible #
            ###################################
        raw_dataset = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir="./cache")
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
        
        return tokenizer(example['sentence'], 
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
        batch_size=8
    )

    eval_dataloader = DataLoader(
        tokenized_validation_data,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=8
    )


    num_train_epochs = args.duration
    num_update_steps_per_epoch = len(train_dataloader)
    cfg.TOT_TRAIN_STEP = num_training_steps = num_train_epochs * num_update_steps_per_epoch
    cfg.PRUNE_END_STEP = len(train_dataloader)*args.prune_end
    cfg.PRUNE_START_STEP = len(train_dataloader)*args.prune_start

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        eps=1e-8
    )

    accelerator = Accelerator(mixed_precision='fp16')
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

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
                outputs = model(batch["input_ids"], labels=batch["input_ids"])

            losses.append(accelerator.gather(outputs.loss))
        loss = torch.mean(torch.cat(losses))
        try:
            perplexity = torch.exp(loss)
        except OverflowError:
            perplexity = float("inf")
        return loss.item(), perplexity.item()
    
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_train_epochs):

        # if args.watch:
        #     watch_quantize_weight(model)
        model.train()
        wandb.log({'epoch': epoch}, commit=False)
        cfg.IS_TRAIN=True

        for step, batch in enumerate(train_dataloader):
            curr_step = len(train_dataloader)*epoch+step
            pruner.prune(curr_step)
            pruner.log_sparsity()
            pruner.monitor_scheduler_step(optimizer)

            outputs = model(**batch)
            loss = outputs.loss
            wandb.log({'Training Loss': loss})
            accelerator.backward(loss)
            pruner.apply_non_prune_gradient(step)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        
        model.eval()
        cfg.IS_TRAIN = False
        eval_loss, eval_ppl = evaluate(model, eval_dataloader)

        wandb.log({'Validation Loss': eval_loss}, commit=False)
        wandb.log({'Validation PPL': eval_ppl}, commit=False)

        accelerator.wait_for_everyone()
        if pruner.cur_sparsity == args.final_sparsity:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.save_folder+"_epoch", save_function=accelerator.save)


if __name__ == "__main__":
    main()

