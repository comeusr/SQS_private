import torch
import math
import re
import warnings
from composer.core import Algorithm, Event

class QUANT_PRUNER(Algorithm):

    def __init__(self, num_components,*args, **kwargs):
        self.num_num_components = num_components


        super().__init__(*args, **kwargs)

    
    def param_initialization(self, init_weight):
        
        return 

        

