import torch
import torch.nn.functional as F
import numpy as np

import wandb

import config as cfg
from QuantAttention import CustomizGPT2Attention, CustomizedQwen2Attention, CustomizedLlamaAttention, CustomizedLLamaMLP

from composer.core import Algorithm

def sigmoid_derivative(x):
    return F.sigmoid(x)*(1-F.sigmoid(x))

class GPT2_PRUNER():

    def __init__(self, model, init_sparsity, final_sparsity, alpha_f):
        self.init_sparsity = init_sparsity
        self.final_sparsity = final_sparsity
        self.cur_sparsity = 0
        self.f_alpha = 0.1
        self.alpha_f = alpha_f
        self.model = model

    def log_mlp_weight(self):
        for name, m in self.model.named_modules():
            if isinstance(m, CustomizedLLamaMLP):
                quantized_up_proj_weights, quantized_down_proj_weights = m.QuantizedWeights()
                up_proj_weight = m.up_proj.weight.data.flatten()
                down_proj_weight = m.down_proj.weight.data.flatten()
                sorted_up_proj_weight = torch.sort(up_proj_weight)[0].cpu().numpy()
                sorted_down_proj_weight = torch.sort(down_proj_weight)[0].cpu().numpy()

                # print("-"*30+"Up proj weight min {}".format(sorted_up_proj_weight[0])+'-'*30) 
                # print("-"*30+"Up proj weight max {}".format(sorted_up_proj_weight[-1])+'-'*30)
                # print("-"*30+"Up proj weight shape {}".format(sorted_up_proj_weight.shape)+'-'*30)

                wandb.log({name+"_up_proj_weight": wandb.Histogram(up_proj_weight.data.cpu().numpy())}, commit=False)
                wandb.log({name+"_down_proj_weight": wandb.Histogram(down_proj_weight.data.cpu().numpy())}, commit=False)
                wandb.log({name+"_quantized_up_proj_weights": wandb.Histogram(quantized_up_proj_weights.data.cpu().numpy())}, commit=False)
                wandb.log({name+"_quantized_down_proj_weights": wandb.Histogram(quantized_down_proj_weights.data.cpu().numpy())}, commit=False)
                
                wandb.log({name+'_up_proj_weight_left_tail': wandb.Histogram(sorted_up_proj_weight[:len(sorted_up_proj_weight)//200])}, commit=False)
                wandb.log({name+'_up_proj_weight_right_tail': wandb.Histogram(sorted_up_proj_weight[-len(sorted_up_proj_weight)//200:])}, commit=False)
                wandb.log({name+'_down_proj_weight_left_tail': wandb.Histogram(sorted_down_proj_weight[:len(sorted_down_proj_weight)//200])}, commit=False)
                wandb.log({name+'_down_proj_weight_right_tail': wandb.Histogram(sorted_down_proj_weight[-len(sorted_down_proj_weight)//200:])}, commit=False)


                # for block_idx in range(m.blocks):
                #     if block_idx == 0 or block_idx == m.blocks - 1:
                #         continue
                #     up_grad = m.up_proj.sub_distribution_list[block_idx].mu.grad
                #     down_grad = m.down_proj.sub_distribution_list[block_idx].mu.grad
                #     wandb.log({name+"_up_proj_weight_{}_grad_norm".format(block_idx): np.linalg.norm(up_grad.data.cpu().numpy())}, commit=False)
                #     wandb.log({name+"_down_proj_weight_{}_grad_norm".format(block_idx): np.linalg.norm(down_grad.data.cpu().numpy())}, commit=False)

                #     wandb.log({name+"_up_proj_weight_{}_grad".format(block_idx): wandb.Histogram(up_grad.data.cpu().numpy())}, commit=False)
                #     wandb.log({name+"_down_proj_weight_{}_grad".format(block_idx): wandb.Histogram(down_grad.data.cpu().numpy())}, commit=False)

    def caculate_mask_thresh(self, model, sparsity):
        # Calculuate the pruning threshold for a given sparsity
        # Smaller Pruning Parameters have higher chance to be pruned
        # Ex: Finial_spasity 0.8, then prune the smallest 80% parameters
        is_dict = {}
        for name, m in model.named_modules():
            if isinstance(m, CustomizGPT2Attention):
                is_dict[name+'.c_attn'] = m.c_attn.sub_distribution.pruning_parameter.detach()
                is_dict[name+'.c_proj'] = m.c_proj.sub_distribution.pruning_parameter.detach()
            elif isinstance(m, (CustomizedQwen2Attention, CustomizedLlamaAttention)):
                is_dict[name+'k_proj'] = m.k_proj.sub_distribution.pruning_parameter.detach()
                is_dict[name+'v_proj'] = m.v_proj.sub_distribution.pruning_parameter.detach()
                is_dict[name+'q_proj'] = m.q_proj.sub_distribution.pruning_parameter.detach()
                is_dict[name+'o_proj'] = m.o_proj.sub_distribution.pruning_parameter.detach()
            elif isinstance(m, CustomizedLLamaMLP):
                for i in range(m.blocks):
                    if i == 0 or i == m.blocks - 1:
                        continue
                    is_dict[name+'.blocks.{}.up_proj'.format(i)] = m.up_proj.sub_distribution_list[i].pruning_parameter.detach()
                    is_dict[name+'.blocks.{}.down_proj'.format(i)] = m.down_proj.sub_distribution_list[i].pruning_parameter.detach()

                # print("is_dict_{} {}".format(name, is_dict[name]))
        
        all_is = torch.cat([is_dict[name].view(-1) for name in is_dict])
        # print("Sparsity {}".format(sparsity))
        # print("All IS {}".format(all_is))
        # print("all_is dimension {}".format(all_is.shape))
        # print("If kth less than total {}".format(int(sparsity*all_is.shape[0]) < all_is.shape[0]))
        # print('K th smallest elemment {}'.format(int(sparsity*all_is.shape[0])))
        mask_thresh = torch.kthvalue(all_is, int(sparsity*all_is.shape[0]))[0].item()
        return mask_thresh, is_dict

    def apply_pruning_grad(self, model):
        
        with torch.no_grad():
            for name, m in model.named_modules():
                if isinstance(m, CustomizGPT2Attention):
                    # print("Applying sparsisty Gradients")
                    sp=0.01
                    attnLayer = m.c_attn.sub_distribution
                    projLayer = m.c_proj.sub_distribution

                    attnP = attnLayer.pruning_parameter/cfg.PRUNE_SCALE
                    attnLayer.pruning_parameter.grad.add_(torch.log(F.sigmoid(attnP)/(sp))*sigmoid_derivative(attnP))

                    projP = projLayer.pruning_parameter/cfg.PRUNE_SCALE
                    projLayer.pruning_parameter.grad.add_(torch.log(F.sigmoid(projP)/(sp))*sigmoid_derivative(projP))

                    # layer.pruning_parameter.grad.add_(torch.log((1-sp)/(1-F.sigmoid(p)))*sigmoid_derivative(p))

                    # attnMu = attnLayer.mu
                    # attnMu.grad.add_(attnMu, alpha=1/(attnLayer.init_sigma ** 2))

                    # projMu = projLayer.mu
                    # projMu.grad.add_(projMu, alpha=1/(projLayer.init_sigma ** 2))
                elif isinstance(m, (CustomizedQwen2Attention, CustomizedLlamaAttention)):
                    sp=0.01
                    k_projLayer = m.k_proj.sub_distribution
                    v_projLayer = m.v_proj.sub_distribution
                    q_projLayer = m.q_proj.sub_distribution
                    o_projLayer = m.o_proj.sub_distribution

                    # k_temp = torch.log(F.sigmoid(k_projP)/(sp))*sigmoid_derivative(k_projP)
                    # if k_temp.isnan().any():
                    #     print('-'*30+'k_temp is nan'+'-'*30)
                    #     print(k_temp)

                    k_projP = k_projLayer.pruning_parameter/cfg.PRUNE_SCALE
                    k_projLayer.pruning_parameter.grad.add_(torch.log(F.sigmoid(k_projP)/(sp))*sigmoid_derivative(k_projP)) 

                    v_projP = v_projLayer.pruning_parameter/cfg.PRUNE_SCALE
                    v_projLayer.pruning_parameter.grad.add_(torch.log(F.sigmoid(v_projP)/(sp))*sigmoid_derivative(v_projP)) 

                    q_projP = q_projLayer.pruning_parameter/cfg.PRUNE_SCALE
                    q_projLayer.pruning_parameter.grad.add_(torch.log(F.sigmoid(q_projP)/(sp))*sigmoid_derivative(q_projP)) 

                    o_projP = o_projLayer.pruning_parameter/cfg.PRUNE_SCALE
                    o_projLayer.pruning_parameter.grad.add_(torch.log(F.sigmoid(o_projP)/(sp))*sigmoid_derivative(o_projP))

                    # k_projMu = k_projLayer.mu
                    # k_projMu.grad.add_(k_projMu/(k_projLayer.init_sigma ** 2))

                    # v_projMu = v_projLayer.mu
                    # v_projMu.grad.add_(v_projMu/(v_projLayer.init_sigma ** 2))

                    # q_projMu = q_projLayer.mu
                    # q_projMu.grad.add_(q_projMu/(q_projLayer.init_sigma ** 2))

                    # o_projMu = o_projLayer.mu
                    # o_projMu.grad.add_(o_projMu/(o_projLayer.init_sigma ** 2))

        return      
    
    def generate_mask(self, model, mask_thresh, is_dict):
        for name, m in model.named_modules():
            if isinstance(m, CustomizGPT2Attention):
                m.c_attn.sub_distribution.mask = (is_dict[name+'.c_attn'] < mask_thresh)
                m.c_proj.sub_distribution.mask = (is_dict[name+'.c_proj'] < mask_thresh)
            elif isinstance(m, (CustomizedQwen2Attention, CustomizedLlamaAttention)):
                m.k_proj.sub_distribution.mask = (is_dict[name+'k_proj'] < mask_thresh)
                m.v_proj.sub_distribution.mask = (is_dict[name+'v_proj'] < mask_thresh)
                m.q_proj.sub_distribution.mask = (is_dict[name+'q_proj'] < mask_thresh)
                m.o_proj.sub_distribution.mask = (is_dict[name+'o_proj'] < mask_thresh)
                # print("Threshold {}".format(mask_thresh))
                # print(m.sub_distribution.mask)
        return 
    
    def sparsity_scheduler(self, train_step):
        if cfg.PRUNE_START_STEP < train_step <= cfg.PRUNE_END_STEP:
            _frac = 1-(train_step-cfg.PRUNE_START_STEP)/(cfg.PRUNE_END_STEP-cfg.PRUNE_START_STEP)
            sparsity = self.final_sparsity + (self.init_sparsity-self.final_sparsity) * (_frac ** 3)
            self.cur_sparsity = sparsity
        else:
            sparsity = self.final_sparsity
            self.cur_sparsity = sparsity
        # print('Fraction {}'.format(_frac))
        return sparsity
    
    def apply_mu_sigma_grad(self, model):
         with torch.no_grad():
            for name, m in model.named_modules():
                if isinstance(m, CustomizGPT2Attention):
                    # print("Applying sparsisty Gradients")
                    attnLayer = m.c_attn.sub_distribution

                    # attnMu = attnLayer.mu
                    # attnMu.grad.add_(attnMu/(attnLayer.init_sigma ** 2))

                    # attnSigma = attnLayer.sigma
                    # attnSigma.grad.add_(attnSigma/(attnLayer.init_sigma ** 2)- 1/attnSigma)

                    # projLayer = m.c_proj.sub_distribution

                    # projMu = projLayer.mu
                    # projMu.grad.add_(projMu/(projLayer.init_sigma ** 2))

                    # projSigma = projLayer.sigma
                    # projSigma.grad.add_(projSigma/(projLayer.init_sigma ** 2)- 1/projSigma)
                elif isinstance(m, (CustomizedQwen2Attention, CustomizedLlamaAttention)):
                    k_projLayer = m.k_proj.sub_distribution
                    v_projLayer = m.v_proj.sub_distribution
                    q_projLayer = m.q_proj.sub_distribution
                    o_projLayer = m.o_proj.sub_distribution

                    # k_projMu = k_projLayer.mu
                    # k_projMu.grad.add_(k_projMu/(k_projLayer.init_sigma ** 2))

                    # k_projSigma = k_projLayer.sigma
                    # k_projSigma.grad.add_(k_projSigma/(k_projLayer.init_sigma ** 2)- 1/k_projSigma) 

                    # v_projMu = v_projLayer.mu
                    # v_projMu.grad.add_(v_projMu/(v_projLayer.init_sigma ** 2))

                    # v_projSigma = v_projLayer.sigma
                    # v_projSigma.grad.add_(v_projSigma/(v_projLayer.init_sigma ** 2)- 1/v_projSigma) 

                    # q_projMu = q_projLayer.mu
                    # q_projMu.grad.add_(q_projMu/(q_projLayer.init_sigma ** 2))

                    # q_projSigma = q_projLayer.sigma
                    # q_projSigma.grad.add_(q_projSigma/(q_projLayer.init_sigma ** 2)- 1/q_projSigma) 

                    # o_projMu = o_projLayer.mu
                    # o_projMu.grad.add_(o_projMu/(o_projLayer.init_sigma ** 2))

                    # o_projSigma = o_projLayer.sigma
                    # o_projSigma.grad.add_(o_projSigma/(o_projLayer.init_sigma ** 2)- 1/o_projSigma) 
    
    def pruning_grad_true(self, model):
        # Set the pruning parameter grad equal to True
        for name, m in model.named_modules():
            if isinstance(m, CustomizGPT2Attention):
                m.c_attn.sub_distribution.pruning_parameter.requires_grad=True
                m.c_proj.sub_distribution.pruning_parameter.requires_grad=True
            elif isinstance(m, (CustomizedQwen2Attention, CustomizedLlamaAttention)):
                m.k_proj.sub_distribution.pruning_parameter.requires_grad=True
                m.v_proj.sub_distribution.pruning_parameter.requires_grad=True
                m.q_proj.sub_distribution.pruning_parameter.requires_grad=True
                m.o_proj.sub_distribution.pruning_parameter.requires_grad=True

    
    def pruning_grad_false(self, model):
        # Set the pruning parameter grad equal to False
        for name, m in model.named_modules():
            if isinstance(m, CustomizGPT2Attention):
                m.c_attn.sub_distribution.pruning_parameter.requires_grad=False
                m.c_proj.sub_distribution.pruning_parameter.requires_grad=False
            elif isinstance(m, (CustomizedQwen2Attention, CustomizedLlamaAttention)):
                m.k_proj.sub_distribution.pruning_parameter.requires_grad=False
                m.v_proj.sub_distribution.pruning_parameter.requires_grad=False
                m.q_proj.sub_distribution.pruning_parameter.requires_grad=False
                m.o_proj.sub_distribution.pruning_parameter.requires_grad=False

    def prune_with_mask(self, model):
        for name, m in model.named_modules():
            if isinstance(m, CustomizGPT2Attention):
                attnMask = m.c_attn.sub_distribution.mask
                m.c_attn.sub_distribution.pruning_parameter.detach().masked_fill_(attnMask, -0.1)
                projMask = m.c_proj.sub_distribution.mask
                m.c_proj.sub_distribution.pruning_parameter.detach().masked_fill_(projMask, -0.1)
            elif isinstance(m, (CustomizedQwen2Attention, CustomizedLlamaAttention)):
                k_projMask = m.k_proj.sub_distribution.mask
                m.k_proj.sub_distribution.pruning_parameter.detach().masked_fill_(k_projMask, -0.1)
                v_projMask = m.v_proj.sub_distribution.mask
                m.v_proj.sub_distribution.pruning_parameter.detach().masked_fill_(v_projMask, -0.1)
                q_projMask = m.q_proj.sub_distribution.mask
                m.q_proj.sub_distribution.pruning_parameter.detach().masked_fill_(q_projMask, -0.1)
                o_projMask = m.o_proj.sub_distribution.mask
                m.o_proj.sub_distribution.pruning_parameter.detach().masked_fill_(o_projMask, -0.1) 



    def monitor_scheduler_step(self, optimizer):
        # print(optimzier)
        for i in range(len(optimizer.param_groups)):
            lr = optimizer.param_groups[i]['lr']
            wandb.log({'parameter_{}_lr'.format(i):lr}, commit=False)

        return
    
    def customize_lr_schduler(self, step):

        # optimizer = state.optimizers[0]
        # for group in optimizer.param_groups:
        #     print(group)
        #     # group['lr'] = group['init_lr']*self.alpha_f*scale
        with torch.no_grad():
            print('Do Nothing')
            # if step >= cfg.PRUNE_END_STEP:

            #     print('Modify Optimizer Learning Rate.')
                
            #     frac = (step-cfg.PRUNE_END_STEP)/(cfg.TOT_TRAIN_STEP-cfg.PRUNE_END_STEP)
            #     scale = self.f_alpha + (1-self.f_alpha)*0.5*(1+math.cos(math.pi*frac))
                
                
                # optimizer = state.optimizers[0]
                # for group in optimizer.param_groups:
                #     group['lr'] = group['initial_lr']*self.alpha_f*scale

        return
    
    # def match(self, event, state):
    #     return event in [Event.BEFORE_TRAIN_BATCH, Event.AFTER_BACKWARD, Event.BATCH_START]
    

    def prune(self, step):

        if cfg.PRUNE and (step <= cfg.PRUNE_START_STEP or step > cfg.PRUNE_END_STEP):
            print('-'*30+'Warming Up phase, not pruning.'+'-'*30)
            self.pruning_grad_false(self.model)
        elif cfg.PRUNE and cfg.PRUNE_START_STEP < step <= cfg.PRUNE_END_STEP:
            print('-'*30+'Pruning phase'+'-'*30)
            # Set Pruning parameter trainable
            self.pruning_grad_true(self.model)
            # Calculate the curr sparsity
            self.sparsity_scheduler(step)
            # Generate mask threshold and help dictionary 
            # is_dict =  {'layer_name': pruning_parameter}
            mask_threshold, is_dict = self.caculate_mask_thresh(self.model, self.cur_sparsity)
            # Generate mask for pruning 
            # mask = {'layer_name': bool matrix}
            self.generate_mask(self.model, mask_threshold, is_dict)
            #Prune with mask
            self.prune_with_mask(self.model)
    
    def apply_non_prune_gradient(self, step):
        if cfg.PRUNE and cfg.PRUNE_START_STEP < step <= cfg.PRUNE_END_STEP:
                print('-'*30+'Apply Pruning Gradient'+'-'*30)
                self.apply_pruning_grad(self.model)
        # try:
        #     if cfg.PRUNE and cfg.PRUNE_START_STEP < step <= cfg.PRUNE_END_STEP:
        #         print('-'*30+'Apply Pruning Gradient'+'-'*30)
        #         self.apply_pruning_grad(self.model)
        #     # elif cfg.PRUNE and (step <= cfg.PRUNE_START_STEP or step > cfg.PRUNE_END_STEP):
        #     #     print('-'*30+'Apply Mu Sigma Gradient'+'-'*30)
        #     #     self.apply_mu_sigma_grad(self.model)
        # except:
        #     prune_in_progress = cfg.PRUNE and cfg.PRUNE_START_STEP < step <= cfg.PRUNE_END_STEP
        #     if prune_in_progress:
        #         print("Pruning in progress.")
        #     else:
        #         print("Not in Pruning.")
            
        #     print("Attention Require Grad or Not")
        #     for name, m in self.model.named_modules():
        #         if isinstance(m, CustomizGPT2Attention):
        #             print(m.c_attn.sub_distribution.pruning_parameter.requires_grad)
        #         elif isinstance(m, (CustomizedQwen2Attention, CustomizedLlamaAttention)):
        #             print(m.k_proj.sub_distribution.pruning_parameter.requires_grad)
        #             print(m.v_proj.sub_distribution.pruning_parameter.requires_grad)
        #             print(m.q_proj.sub_distribution.pruning_parameter.requires_grad)
        #             print(m.o_proj.sub_distribution.pruning_parameter.requires_grad)

    
    def log_sparsity(self):
        wandb.log({'Sparsity': self.cur_sparsity}, commit=False)

    
    # def apply(self, event, state, logger):
    #     train_step = state.timestamp.batch.value
    #     # if cfg.PRUNE and cfg.PRUNE_START_STEP > 0:
    #     # TO DO
    #     # Apply the KL-divergence gradients
    #     if event == Event.BEFORE_TRAIN_BATCH:
    #         # Prune the parameter according to the pruning parameters
    #         if cfg.PRUNE and (train_step <= cfg.PRUNE_START_STEP or train_step > cfg.PRUNE_END_STEP):
    #             self.pruning_grad_false(state.model)
    #         elif cfg.PRUNE and cfg.PRUNE_START_STEP < train_step <= cfg.PRUNE_END_STEP:
    #             # Set Pruning parameter trainable
    #             self.pruning_grad_true(state.model)
    #             # Calculate the curr sparsity
    #             self.sparsity_scheduler(train_step)
    #             # Generate mask threshold and help dictionary 
    #             # is_dict =  {'layer_name': pruning_parameter}
    #             mask_threshold, is_dict = self.caculate_mask_thresh(state.model, self.cur_sparsity)
    #             # Generate mask for pruning 
    #             # mask = {'layer_name': bool matrix}
    #             self.generate_mask(state.model, mask_threshold, is_dict)
    #             #Prune with mask
    #             self.prune_with_mask(state.model)
            
                            
    #     elif event == Event.AFTER_BACKWARD:
    #         # Add the gradients of KL divergence to pruning parameters
    #         # print("Apply Pruning Gradient")
    #         if cfg.PRUNE and cfg.PRUNE_START_STEP < train_step <= cfg.PRUNE_END_STEP:
    #             self.apply_pruning_grad(state.model)
    #         elif cfg.PRUNE and (train_step <= cfg.PRUNE_START_STEP or train_step > cfg.PRUNE_END_STEP):
    #             self.apply_mu_sigma_grad(state.model)
    #     elif event == event.BATCH_START:
    #         logger.log_metrics({'sparsity': self.cur_sparsity})

    #     return
    