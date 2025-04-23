
import torch
from SQS.utils.utils import get_distribution

def interval_mapping(M: torch.tensor, B:int, DEVICE):
    Mvect= M.view(-1)
    quantiles = torch.quantile(Mvect, torch.linspace(0, 1, steps=B).cuda())
    # Assign each element to a bin index
    bin_indices = torch.bucketize(Mvect, quantiles, right=False)  # Adjust to 0-based index
    M.to(DEVICE)

    sum_per_bin   = torch.zeros(B, device=Mvect.device).scatter_add_(0, bin_indices, Mvect)
    count_per_bin = torch.zeros(B, device=Mvect.device).scatter_add_(0, bin_indices,
                                                                 torch.ones_like(Mvect))
    # --- keep only the non-empty bins -------------------
    means = torch.zeros(B, device=Mvect.device)
    valid = count_per_bin != 0
    means[valid] = sum_per_bin[valid] / count_per_bin[valid]  

    if B != len(valid):
        print("-"*50+"Interval Mapping B {}".format(B)+"-"*50)
        print("-"*50+"Interval Mapping len valid {}".format(len(valid))+"-"*50)
        print("-"*50+"Interval Mapping means shape{}".format(means.shape)+"-"*50)

    has_nan = torch.isnan(means).any()
    if has_nan:
        print("-"*50+"Interval Mapping found nan in means {}".format(means)+"-"*50)

    return bin_indices.cpu(), means.to(DEVICE)



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

def reconstruct(dims, Ws, bin_indices, device):

    # Ws: B x 1
    # bin_indices: N x 1

    N = dims[0]*dims[1] if len(dims) == 2 else dims[0]

    unique_bin_indices = torch.unique(bin_indices)
    # Create an empty matrix of the given dimensions
    M = torch.zeros(N, dtype=Ws.dtype, device=Ws.device)

    # Efficiently assign values using advanced indexing
    for i in unique_bin_indices:
        M[bin_indices == i] = Ws[i]

    return M.to(device)

