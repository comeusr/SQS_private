import torch.nn as nn
from QuantAttention import CustomizeBertSelfAttention


def InitBertModel(model:nn.Module, sigma):

    for name, m in model.named_modules():
        if isinstance(m, CustomizeBertSelfAttention):
            m.init_mask_params(sigma)




    