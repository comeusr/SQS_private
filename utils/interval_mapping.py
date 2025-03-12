
import torch
from collections import OrderedDict
from utils import get_distribution



def interval_mapping_and_reconstruct(M: torch.tensor, Pm: torch.tensor, B:int, K: int, pi_normalized, sigma, DEVICE):
    # M: high-dim matrix or vetor 
    # Pm: a vector of 16 different values
    # Bucket
    dims = M.size()
    #### flatten the input matrix
    Mvect= M.view(-1)
    quantiles = torch.quantile(Mvect, torch.linspace(0, 1, steps=B))
    # Assign each element to a bin index
    bin_indices = torch.bucketize(Mvect, quantiles, right=False)  # Adjust to 0-based index
    
    # Compute mean value for each bin
    means = torch.zeros(B, dtype=torch.float32)
    for i in range(B):
        means[i] = Mvect[bin_indices == i].mean()

    # create a matrix of dimension B x 16
    O = get_distribution(means, Pm, K, pi_normalized, sigma, DEVICE)
    # O: B x 16
    # Pm: 16 x 1
    Ws = O @ Pm
    # Ws: [B, 1]
    recon_M_vect=reconstruct(dims, Ws, bin_indices)
    recon_M = recon_M_vect.size(dims)
    return recon_M

def reconstruct(dims, Ws, bin_indices):
    # Create an empty matrix of the given dimensions
    M = torch.zeros(dims, dtype=Ws.dtype, device=Ws.device)

    # Efficiently assign values using advanced indexing
    M[torch.arange(len(bin_indices))] = Ws[bin_indices]

    return M

