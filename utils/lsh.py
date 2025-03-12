import numpy as np
from sklearn.random_projection import GaussianRandomProjection
import torch
from collections import OrderedDict
from utils.utils import get_distribution

def LSH(vector: torch.tensor, B, DEVICE):
    vmin = torch.min(vector)
    vmax = torch.max(vector)

    thresholds = vmin+(vmax-vmin)*torch.rand(B)

    binary_codes = vector.unsqueeze(1) <= thresholds.unsqueeze(0)
    unique_codes, inverse_indices = torch.unique(binary_codes, dim=0, return_inverse=True)
    # print("Unique codes", unique_codes)
    print("Inverse indices Length", len(inverse_indices))
    
    # Create buckets: mapping from binary code (as a tuple) to list of (index, value)
    buckets = OrderedDict()
    for code_idx, code in enumerate(unique_codes):
        # Find indices for which the binary code equals the current unique code
        indices = (inverse_indices == code_idx).nonzero(as_tuple=True)[0]
        # Convert the binary code tensor to a tuple of booleans to use as a key.
        buckets[tuple(code.tolist())] = list(zip(indices.tolist(), vector[indices].tolist()))

    
    
    # Compute mean values for each bucket
    Nums = torch.empty(len(buckets))
    for i, items in enumerate(buckets.values()):
        # Extract the values from the (index, value) tuples
        values = torch.tensor([item[1] for item in items])
        Nums[i] = torch.mean(values)

    print("Num size", Nums.size())
    
    return buckets, Nums.to(DEVICE)


def LSH_bucketing_and_reconstruct(M: torch.tensor, Pm: torch.tensor, B:int, K: int, pi_normalized, sigma, DEVICE):
    # M: high-dim matrix or vetor 
    # Pm: a vector of 16 different values
    # Bucket

    dims = M.size()
    #### flatten the input matrix
    Mvect= M.view(-1)
    invMap, nums=LSH(Mvect, B)
    
    # create a matrix of dimension B x 16
    O = get_distribution(nums, Pm, K, pi_normalized, sigma, DEVICE)
    Ws = O @ Pm
    recon_M_vect=reconstruct(dims, Ws, invMap)
    recon_M = recon_M_vect.size(dims)
    return recon_M

def reconstruct(dims, Ws, invMAP):
    # dimension of input matrix M
    # Ws: weighted sum
    # inverse map from bucket to index in matrix M
    B = len(invMAP)

    M = torch.zeros(dims)
    for bi, one_bucket in enumerate(invMAP):
        for (index, _) in one_bucket:
            M[index] = Ws[bi]
    return M 
