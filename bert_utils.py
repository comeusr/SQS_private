import torch.nn as nn
from QuantAttention import CustomizeBertSelfAttention
import torch
from collections.abc import Mapping
import numpy as np
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

InputDataClass = NewType("InputDataClass", Any)


def InitBertModel(model:nn.Module, sigma):

    for name, m in model.named_modules():
        if isinstance(m, CustomizeBertSelfAttention):
            m.init_mask_params(sigma)


def torch_customized_data_collator(features: List[InputDataClass]) -> Dict[str, Any]:
    
    # print('Take a look at features first Time')
    # print(features)
    # print('-'*66)

    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.


    print("First Feature")
    print(features[0].keys())

    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features if f is not None])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features if f is not None]))
            else:
                batch[k] = torch.tensor([f[k] for f in features if f is not None])

    return batch


def customized_data_collator(features: List[InputDataClass], return_tensors="pt") -> Dict[str, Any]:
    if  return_tensors=="pt":
        return torch_customized_data_collator(features)
