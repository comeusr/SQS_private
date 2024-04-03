import torch

x = torch.tensor([0.001, 0.000005, 0.0, 0.000000])
print(x.eq(0.0).float().sum())