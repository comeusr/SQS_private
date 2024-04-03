import wandb
from composer import Callback, State, Logger, Event
from modeling.DGMS import DGMSConv
import torch.nn as nn
import torch

def check_total_zero(x):
    with torch.no_grad():
        return x.eq(0.0).float().sum().item()

def check_total_weights(x):
    with torch.no_grad():
        return x.numel()

class Sparsity(Callback):

    def __init__(self):
        super().__init__()

    def log_mu_sparsity(self, state:State, event:Event, logger:Logger):
        for name, m in state.model.named_modules():
            if isinstance(m, DGMSConv):
                wandb.log({name+"mu": wandb.Histogram(m.sub_distribution.mu.detach().data)}, commit=False)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                total_zero = check_total_zero(m.weight)
                total_weight = check_total_weights(m.weight)
                wandb.log({name+"sparsity": total_zero/total_weight})


