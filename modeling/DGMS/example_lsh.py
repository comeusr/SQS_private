import numpy as np
from sklearn.random_projection import GaussianRandomProjection
import torch
from collections import OrderedDict
from utils import get_distribution

def LSH(vector: torch.tensor, B):
    thresholds=torch.random(min=torch.min(vector), max=torch.max(vector), size=B)
    Nums = torch.zeros(B)
    Buckets=OrderedDict()
    for i, vi in enumerate(vector):
        binary_code=[]
        for ti in thresholds:
            binary_code.append(vi <= ti)
        if tuple(binary_code) not in Buckets:
            Buckets[tuple(binary_code)]=[(i, vi)]
        else:
            Buckets[tuple(binary_code)].append((i, vi))
    for i, l in enumerate(Buckets.values()):
        if len(l) > 0:
            Nums[i] = torch.mean(l)
          
    return Buckets, Nums


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

