from __future__ import print_function

import torch.nn as nn
from composer.models import ComposerModel

from .networks import get_network
from modeling.DGMS import DGMSConv
import torch.nn.functional as F
import torchmetrics

class DGMSNet(ComposerModel):
    def __init__(self, network, args, freeze_bn=False):
        super(DGMSNet, self).__init__()
        self.args = args
        self.network = network
        self.freeze_bn = freeze_bn
        self.train_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=args.num_classes, average='micro')
        self.val_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=args.num_classes, average='micro')

    def init_mask_params(self):
        print("--> Start to initialize sub-distribution parameters, this may take some time...")
        for name, m in self.network.named_modules():
            if isinstance(m, DGMSConv):
                m.init_mask_params()
                # m.weight.data.xavier_uniform()

        print("--> Sub-distribution parameters initialization finished!")

    def forward(self, batch):
        inputs, _ = batch
        return self.network(inputs)

    def eval_forward(self, batch, outputs):
        if outputs:
            return outputs
        inputs, _ = batch
        return self.network(inputs)

    def update_metric(self, batch, outputs, metric):
        _, targets = batch
        metric.update(outputs, targets)

    def get_metrics(self, is_train=False):
        # defines which metrics to use in each phase of training
        return {'MulticlassAccuracy': self.train_accuracy} if is_train else {'MulticlassAccuracy': self.val_accuracy}

    def get_1x_lr_params(self):
        self.init_mask_params()
        modules = [self.network]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        if self.args.freeze_weights:
                            for p in m[1].parameters():
                                pass
                        else:
                            for p in m[1].parameters():
                                if p.requires_grad:
                                    yield p
                else:
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def loss(self, outputs, batch):
        _, targets = batch
        return F.cross_entropy(outputs, targets)

