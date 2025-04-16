import wandb
from composer import Callback, State, Logger, Event
from modeling.DGMS import DGMSConv
import torch.nn as nn
import torch
import numpy as np
from bert_utils import CustomizeBertSelfAttention


def check_total_zero(x):
    with torch.no_grad():
        return x.eq(0.0).float().sum().item()


def check_total_weights(x):
    with torch.no_grad():
        return x.numel()


class EpochMonitor(Callback):

    def run_event(self, event: Event, state: State, logger: Logger):
        if event == Event.FIT_START:
            for name, m in state.model.named_modules():
                if isinstance(m, CustomizeBertSelfAttention):
                    with torch.no_grad():
                        query_mu, key_mu,value_mu = m.getMu()
                        querySweight, keySweight, valueSweight = m.get_Sweight()
                        queryPweight, keyPweight, valuePweight = m.get_Pweight()

                    wandb.log({name+"_S_query": wandb.Histogram(querySweight.data.cpu().numpy())}, commit=False)
                    wandb.log({name+"_S_key": wandb.Histogram(keySweight.data.cpu().numpy())}, commit=False)
                    wandb.log({name+"_S_value": wandb.Histogram(valueSweight.data.cpu().numpy())}, commit=False)


                    wandb.log({name+"_Mu_query": wandb.Histogram(query_mu)}, commit=False)
                    wandb.log({name+"_Mu_key": wandb.Histogram(key_mu)}, commit=False)
                    wandb.log({name+"_Mu_value": wandb.Histogram(value_mu)}, commit=False)


        elif event == Event.EPOCH_END:
            print(f'Epoch: {state.timestamp.epoch}')
            tot_numel = 0
            tot_zero_numel = 0
            for name, m in state.model.named_modules():
                if isinstance(m, DGMSConv):
                    with torch.no_grad():
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
                        tot_numel += P_weight_tot
                        tot_zero_numel += P_weight_zeros

                        wandb.log({name+"_mu": wandb.Histogram(data)}, commit=False)
                        # wandb.log({name+"_P_weight": wandb.Histogram(P_weight.data.cpu().numpy())})
                        # wandb.log({name+"_S_weight": wandb.Histogram(S_weight.data.cpu().numpy())})
                        # wandb.log({name+"_Origin_weight": wandb.Histogram(Origin_weight.data.cpu().numpy())})
                        
                        wandb.log({name+"_P_zeros": P_weight_zeros/P_weight_tot}, commit=False)
                        wandb.log({name+"_S_zeros": S_weight_zeros/S_weight_tot}, commit=False)
                        # wandb.log({name+"_Origin_zeros": Origin_weight_zeros/Origin_weight_tot}, commit=False)
                        # wandb.log({name+"_Pretrained_weight": wandb.Histogram(Origin_weight.cpu().numpy())}, commit=False)
                        # wandb.log({name+"_Soft_weight": wandb.Histogram(S_weight.cpu().numpy())}, commit=False)

                        # if name == 'network.layer1.0.conv1':
                        #     # print("network.layer1.0.conv1 {}".format(S_weight))
                        #     print("Region Belonging shape {}".format(m.sub_distribution.region_belonging.shape))
                        #     print("Region Belonging 0 {}".format(m.sub_distribution.region_belonging[0]))
                        #     print("Region Belonging 1 {}".format(m.sub_distribution.region_belonging[1]))
                        #     print("Region Belonging first row zero numel {}".format(m.sub_distribution.region_belonging[0].eq(0.0).sum()))
                        #     print("Region Belonging second row zero numel {}".format(m.sub_distribution.region_belonging[1].eq(0.0).sum()))
                    wandb.log({'Total Sparsity': tot_zero_numel/tot_numel}, commit=False)
                elif isinstance(m, CustomizeBertSelfAttention):
                    with torch.no_grad():
                        query_mu, key_mu,value_mu = m.getMu()
                        querySweight, keySweight, valueSweight = m.get_Sweight()
                        queryPweight, keyPweight, valuePweight = m.get_Pweight()


                    

                elif isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    total_zero = check_total_zero(m.weight)
                    total_weight = check_total_weights(m.weight)
                    wandb.log({name+"_sparsity": total_zero/total_weight}, commit=False)
        # elif event == event.BATCH_START and state.timestamp.batch.value % 5 == 0:
            


