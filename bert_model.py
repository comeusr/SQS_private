from typing import Union, Optional, Any

import torch

import transformers
from torchmetrics import Metric

from composer.models.base import ComposerModel
from composer.models import HuggingFaceModel

import config as cfg

import transformers
from peft import PeftConfig, PeftModel
from transformers import PretrainedConfig
from transformers.models.auto.auto_factory import _BaseAutoModelClass

class customize_hf_model(HuggingFaceModel):

    def __init__(self,
                model: Union[transformers.PreTrainedModel, 'PeftModel'],
                tokenizer: Optional[Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]] = None,
                use_logits: Optional[bool] = False,
                metrics: Optional[list[Metric]] = None,
                eval_metrics: Optional[list[Metric]] = None,
                shift_labels: Optional[bool] = None,
                allow_embedding_resizing: bool = False,
                peft_config: Optional['PeftConfig'] = None,
                should_save_peft_only: bool = True,
            )-> None:
        
        super.__init__(
            model=model,
            tokenizer=tokenizer,
            use_logits=use_logits,
            metrics=metrics,
            eval_metrics=eval_metrics,
            shift_labels=shift_labels,
            allow_embedding_resizing=allow_embedding_resizing,
            peft_config=peft_config,
            should_save_peft_only=should_save_peft_only)
        

    def eval_forward(self, batch, outputs: Optional[Any] = None):
        cfg.IS_TRAIN = False

        outputs = outputs if outputs else self.model.forward(batch)

        
        
        
        
        
