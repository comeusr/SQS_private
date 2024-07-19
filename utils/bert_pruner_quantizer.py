from composer.core import Algorithm, Event
from composer.models import ComposerModel
from composer import State
from modeling.DGMS import DGMSConv
import config as cfg
import torch
import math
import torch.nn.functional as F