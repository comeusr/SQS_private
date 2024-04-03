import wandb
from composer import Callback, State, Logger, Event
from modeling.DGMS import DGMSConv
import torch.nn as nn
import torch
import numpy as np


def check_total_zero(x):
    with torch.no_grad():
        return x.eq(0.0).float().sum().item()


def check_total_weights(x):
    with torch.no_grad():
        return x.numel()


class Sparsity(Callback):

    def run_event(self, event:Event, state:State, logger:Logger):
        if event == Event.EPOCH_START:
            print("Try to print Module Name")
            for name, m in state.model.named_modules():
                print(name)
            # if isinstance(m, DGMSConv):
            #     logger.log({name+"mu": wandb.Histogram(m.sub_distribution.mu)})
            # elif isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            #     total_zero = check_total_zero(m.weight)
            #     total_weight = check_total_weights(m.weight)
            #     logger.log({name+"sparsity": total_zero/total_weight})


class EpochMonitor(Callback):

    def run_event(self, event: Event, state: State, logger: Logger):
        if event == Event.EPOCH_START:
            print(f'Epoch: {state.timestamp.epoch}')
            for name, m in state.model.named_modules():
                if isinstance(m, DGMSConv):
                    data = m.sub_distribution.mu.detach().data.cpu().numpy()
                    P_weight = m.get_Pweight()
                    S_weight = m.get_Sweight()
                    Origin_weight = m.weight
                    P_weight_zeros = check_total_zero(P_weight)
                    P_weight_tot = check_total_weights(P_weight)
                    S_weight_zeros = check_total_zero(S_weight)
                    S_weight_tot = check_total_weights(S_weight)
                    Origin_weight_zeros=check_total_zero(Origin_weight)
                    Origin_weight_tot=check_total_weights(Origin_weight)
                    # hist = np.histogram(data)
                    # print(hist)
                    min_mu = min(np.abs(data))
                    wandb.log({name+"min_abs_mu": min_mu})

                    wandb.log({name+"_mu": wandb.Histogram(data)}, commit=False)
                    wandb.log({name+"_P_weight": wandb.Histogram(P_weight.data.cpu().numpy())})
                    wandb.log({name+"_S_weight": wandb.Histogram(S_weight.data.cpu().numpy())})
                    wandb.log({name+"_Origin_weight": wandb.Histogram(Origin_weight.data.cpu().numpy())})

                    wandb.log({name+"_P_zeros": P_weight_zeros/P_weight_tot})
                    wandb.log({name+"_S_zeros": S_weight_zeros/S_weight_tot})
                    wandb.log({name+"_Origin_zeros": Origin_weight_zeros/Origin_weight_tot})


                elif isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    total_zero = check_total_zero(m.weight)
                    total_weight = check_total_weights(m.weight)
                    wandb.log({name+"_sparsity": total_zero/total_weight})

