# DGMS GM Sub-distribution implementation.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config as cfg

from utils.misc import cluster_weights, get_device, cluster_weights_sparsity
from utils.utils import get_distribution
from utils.interval_mapping import interval_mapping, reconstruct
import time

DEVICE = get_device()


def gumbel_max_sample(logits: torch.Tensor) -> torch.Tensor:
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))  # Gumbel(0,1) noise
    return torch.argmax(logits + gumbel_noise, dim=-1)



class GaussianMixtureModel(nn.Module):
    """Concrete GMM for sub-distribution approximation.
    """
    def __init__(self, num_components, init_weights, temperature=0.01, B=11, init_method="k-means", init_sigma=3):
        super(GaussianMixtureModel, self).__init__()
        self.num_components = num_components
        self.temperature = temperature

        # print("Initializing GMM Parameters.")
        shape = init_weights.shape
        print("Init_method", init_method)
        self.params_initialization(init_weights, init_method)
        self.prune = cfg.PRUNE
        self.mask = (init_weights.abs()< 0.0).to(DEVICE)
        
        # print('GMM weight dim {}'.format(init_weights.shape))
        # print('Init Mask dim {}'.format(self.mask.shape))
        if cfg.PRUNE:
            self.init_sigma = init_sigma
        # Reduced dimension
        self.B = B

        
        start_time = time.time()
        print("Starting Grouping")
        self.bin_indices, self.nums = interval_mapping(init_weights, self.B, DEVICE)
        end_time = time.time()
        print("Grouping completed in {} seconds".format(end_time - start_time))

    def params_initialization(self, init_weights, method='k-means'):
        if not cfg.PRUNE:
            """ Initialization of GMM parameters using k-means algorithm. """
            self.mu_zero = torch.tensor([0.0], device=DEVICE).float()
            self.pi_k, self.mu, self.sigma = \
                    torch.ones(self.num_components-1, device=DEVICE), \
                    torch.ones(self.num_components-1, device=DEVICE), \
                    torch.ones(self.num_components-1, device=DEVICE)
            if method == 'k-means':
                print("Using k-means for GMM initialization")
                initial_region_saliency, pi_init, pi_zero_init, sigma_init, _sigma_zero = cluster_weights(init_weights, self.num_components)
            elif method == "quantile":
                initial_region_saliency, pi_init, pi_zero_init, sigma_init, _sigma_zero = cluster_weights(init_weights, self.num_components)
            elif method == 'empirical':
                initial_region_saliency, pi_init, pi_zero_init, sigma_init, _sigma_zero = cluster_weights(init_weights, self.num_components)
                sigma_init, _sigma_zero = torch.ones_like(sigma_init).mul(0.01).to(DEVICE), torch.ones_like(torch.tensor([_sigma_zero])).mul(0.01).to(DEVICE)
            self.mu = nn.Parameter(data=torch.mul(self.mu.to(DEVICE), initial_region_saliency.flatten().to(DEVICE)))
            self.pi_k = nn.Parameter(data=torch.mul(self.pi_k.to(DEVICE), pi_init)).to(DEVICE).float()
            self.pi_zero = nn.Parameter(data=torch.tensor([pi_zero_init], device=DEVICE)).to(DEVICE).float()
            self.sigma_zero = nn.Parameter(data=torch.tensor([_sigma_zero], device=DEVICE)).float()
            self.sigma = nn.Parameter(data=torch.mul(self.sigma, sigma_init)).to(DEVICE).float()
            self.temperature = nn.Parameter(data=torch.tensor([self.temperature], device=DEVICE), requires_grad=False)
        else:
            """ Intialization of GMM + Pruning parameters using k-means"""
            self.mu_zero = torch.tensor([0.0], device=DEVICE).float()
            self.pi_k, self.mu, self.sigma = \
                    torch.ones(self.num_components, device=DEVICE), \
                    torch.ones(self.num_components, device=DEVICE), \
                    torch.ones(self.num_components, device=DEVICE)
            print("Method", method)
            if method == 'k-means':
                initial_region_saliency, pi_init, sigma_init = cluster_weights_sparsity(init_weights, self.num_components)
            elif method == "quantile":
                initial_region_saliency, pi_init, sigma_init = cluster_weights_sparsity(init_weights, self.num_components)
            elif method == 'empirical':
                initial_region_saliency, pi_init, sigma_init = cluster_weights_sparsity(init_weights, self.num_components)
                sigma_init = torch.ones_like(sigma_init).mul(0.01).to(DEVICE)
                
            if sigma_init.isnan().any() or sigma_init.eq(0.0).any():
                print('-'*50+"Sigma init is nan or zero"+"-"*50)
                print('-'*50+"Old Sigma init {}".format(sigma_init)+"-"*50)
                # Create a mask for valid elements (nonzero and non-NaN)
                valid_mask = (sigma_init != 0) & (~torch.isnan(sigma_init)) & (torch.isfinite(sigma_init))

                # Extract valid elements
                valid_elements = sigma_init[valid_mask]
                
                smallest_valid = torch.min(valid_elements)
                largest_valid = torch.max(valid_elements)
                # Replace zeros and NaNs with the smallest valid elements
                sigma_init[sigma_init == 0] = smallest_valid
                sigma_init[torch.isnan(sigma_init)] = smallest_valid
                sigma_init[~torch.isfinite(sigma_init)] = largest_valid
                print('-'*50+"New Sigma init {}".format(sigma_init)+"-"*50)
           
            self.mu = nn.Parameter(data=torch.mul(self.mu.to(DEVICE), initial_region_saliency.flatten().to(DEVICE)))
            self.pi_k = nn.Parameter(data=torch.mul(self.pi_k.to(DEVICE), pi_init)).to(DEVICE).float()
            self.sigma = nn.Parameter(data=torch.mul(self.sigma, sigma_init)).to(DEVICE).float()
            # print("Initial Self Sigma contains zero {}".format(self.sigma.eq(0.0).any()))
            self.temperature = nn.Parameter(data=torch.tensor([self.temperature], device=DEVICE), requires_grad=False)
            self.pruning_parameter = nn.Parameter(data=5*cfg.PRUNE_SCALE*torch.ones_like(init_weights, device=DEVICE))

    def gaussian_mixing_regularization(self):
        
        if not cfg.PRUNE:
            pi_tmp = torch.cat([self.pi_zero, self.pi_k], dim=-1).abs()
            res = torch.div(pi_tmp, pi_tmp.sum(dim=-1).unsqueeze(-1)).to(DEVICE)
            # print('Pi shape{}'.format(res.shape))
            return torch.div(pi_tmp, pi_tmp.sum(dim=-1).unsqueeze(-1)).to(DEVICE)
        else:
            pi_tmp = self.pi_k.abs()
            return torch.div(pi_tmp, pi_tmp.sum(dim=-1).unsqueeze(-1)).to(DEVICE)

    def Normal_pdf(self, x, _pi, mu, sigma):
        """ Standard Normal Distribution PDF. """
        pow2 = torch.pow(x - mu, 2)
        
        sigma = sigma.to(torch.float32)
        pdf = torch.mul(torch.reciprocal(torch.sqrt(torch.mul( \
                torch.tensor([2 * math.pi], device=DEVICE), (sigma**2)))), \
                    torch.exp(-torch.div(pow2, 2 * sigma**2))).mul(_pi)
        if pdf.isnan().any():
            temp = torch.exp(-torch.div(pow2, 2 * sigma**2)-torch.log(torch.sqrt(2*math.pi*sigma**2)))
            # print("Temp all zero {}".format(temp.sum() == 0))
            temp_1 = torch.div(pow2, 2 * sigma**2)
            temp_2 = torch.log(torch.sqrt(2*math.pi*sigma**2))
            # print("X-mu Squared {}".format(torch.pow(x - mu, 2)))
            # print("Sigma Squared {}".format(sigma**2))
            
            print("Pow 2 {}".format(pow2))
            print("Sigma {}".format(sigma))
            print("Sigam Dtype {}".format(sigma.dtype))
            print("Sigma Squared {}".format(sigma**2))

        return pdf


    def GMM_region_responsibility(self, weights):
        if not cfg.PRUNE:
            """" Region responsibility of GMM. """
            pi_normalized = self.gaussian_mixing_regularization().to(DEVICE)
            responsibility = torch.zeros([self.num_components, weights.size(0)], device=DEVICE)
            responsibility[0] = self.Normal_pdf(weights.to(DEVICE), pi_normalized[0], 0.0, self.sigma_zero.to(DEVICE))
            for k in range(self.num_components-1):
                responsibility[k+1] = self.Normal_pdf(weights, pi_normalized[k+1], self.mu[k].to(DEVICE), self.sigma[k].to(DEVICE))
            responsibility = torch.div(responsibility, responsibility.sum(dim=0) + cfg.EPS)
            return F.softmax(responsibility / self.temperature, dim=0)
        else:
            """" Region responsibility of GMM. """

            pi_normalized = self.gaussian_mixing_regularization().to(DEVICE)
            O = get_distribution(self.nums, self.mu, self.num_components, pi_normalized, self.sigma, DEVICE)

            O = torch.div(O, O.sum(dim=0) + cfg.EPS)
            # print("responsibility {}".format(responsibility))
            temp = F.softmax(O / self.temperature, dim=0).T

            # print("-"*50+"Responsibility before mask"+"-"*50)
            
            if temp.isnan().any():
                print('-'*50+"Found nan in the soft weights"+"-"*50)
                if responsibility.isnan().any():
                    print("-"*50+"responsibility is nan"+"-"*50)
                if pi_normalized.isnan().any():
                    print("-"*50+"pi_normalized is nan"+"-"*50)
                if weights.isnan().any():
                    print("-"*50+"weights is nan"+"-"*50)
                if self.mu.isnan().any():
                    print('-'*50+"Mu is nan"+"-"*50)

            return temp


    def forward(self, weights, train=True):
        if not cfg.PRUNE:
            if train:
                # soft mask generalized pruning during training
                self.region_belonging = self.GMM_region_responsibility(weights.flatten())
                # print("Printing the region_belong shape {}".format(self.region_belonging.shape))
                Sweight = torch.mul(self.region_belonging[0], 0.) \
                        + torch.mul(self.region_belonging[1:], self.mu.unsqueeze(1)).sum(dim=0)
                return Sweight.view(weights.size())
            else:
                self.region_belonging = self.GMM_region_responsibility(weights.flatten())
                # print("Printing the region_belong shape {}".format(self.region_belonging.shape))
                max_index = torch.argmax(self.region_belonging, dim=0).unsqueeze(0)
                mask_w = torch.zeros_like(self.region_belonging).scatter_(dim=0, index=max_index, value=1.)
                Pweight = torch.mul(mask_w[1:], self.mu.unsqueeze(1)).sum(dim=0)
                return Pweight.view(weights.size())
        else:
            if train:
                region_belonging = self.GMM_region_responsibility(weights.flatten())
                # print("Printing the region_belong shape {}".format(region_belonging.shape))

                def memory_in_mb(tensor):
                    return tensor.element_size() * tensor.numel() / (1024*1024)
                                
                if cfg.PRIOR == 'spike_slab':
                    Sweight =  reconstruct(weights.size(), region_belonging@self.mu, self.bin_indices)* F.sigmoid(self.pruning_parameter.flatten()/cfg.PRUNE_SCALE)
                else:
                    Sweight = torch.mul(region_belonging, self.mu.unsqueeze(1)).sum(dim=0)* F.sigmoid(self.pruning_parameter.flatten()/cfg.PRUNE_SCALE) \
                            + (1-F.sigmoid(self.pruning_parameter.flatten()/cfg.PRUNE_SCALE))*torch.randn_like(weights.flatten())

                # print("-"*50+"Sweight before delete"+"-"*50)
                torch.cuda.empty_cache()
                # print("-"*50+"Sweight after delete"+"-"*50)

                return Sweight.view(weights.size())
            else:
                region_belonging = self.GMM_region_responsibility(weights.flatten())
                # print("Region belonging shape", region_belonging.shape)
                # print("Printing the region_belong shape {}".format(self.region_belonging.shape))  
           
                if cfg.SAMPLE:
                    # max_index = torch.argmax(self.region_belonging, dim=0).unsqueeze(0)
                    max_index = region_belonging.transpose(0, 1).multinomial(num_samples=1).transpose(0, 1)
                else:
                    max_index = torch.argmax(region_belonging, dim=0).unsqueeze(0)
                # print("Print the max_index shape {}".format(max_index.shape))
                mask_w = torch.zeros_like(region_belonging).scatter_(dim=0, index=max_index, value=1.)
                # print("Region belonging shape", region_belonging.shape)
                # print("Mu shape", self.mu.shape)
                Pweight = reconstruct(weights.size(),region_belonging@self.mu, self.bin_indices)
                
                # print('Pweight before mask {}'.format(Pweight))
                # print("Pweight shape", Pweight.shape)
                Pweight = Pweight.view(weights.size())

                Pweight.detach().masked_fill_(self.mask, 0.0)
                return Pweight

def gmm_approximation(num_components, init_weights, temperature=0.5, B=11, init_method='k-means', sigma=3) -> GaussianMixtureModel:
    return GaussianMixtureModel(num_components, init_weights, temperature, B, init_method, sigma)