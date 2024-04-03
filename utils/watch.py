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
            print("Try to print Module Name")
            for name, m in state.model.named_modules():
                if isinstance(m, DGMSConv):
                    print("Found DGMSConv")
                    data = m.sub_distribution.mu.detach().data.cpu().numpy()
                    hist = np.histogram(data)
                    print(hist)
                    wandb.log({name+"mu": wandb.Histogram(np_histogram=hist)}, step=int(state.timestamp.epoch))
                    print("Logged Histogram")
                elif isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    print("Found Non-DGMS Layer")
                    total_zero = check_total_zero(m.weight)
                    total_weight = check_total_weights(m.weight)
                    print(total_zero/total_weight)
                    wandb.log({name+"sparsity": total_zero/total_weight}, step=int(state.timestamp.epoch))

