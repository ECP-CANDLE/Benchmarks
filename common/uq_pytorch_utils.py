from __future__ import absolute_import

import torch
import torch.nn as nn

def abstention_loss(torch.nn.Module):

    def __init__(self, alpha, mask):
        super(abstention_loss,self).__init__()
        self.alpha = alpha
        self.mask = mask
        self.ndevices = torch.cuda.device_count()
        self.eps = torch.finfo(torch.float32).eps


    def forward(self, y_pred, y_true):

        if self.ndevices > 0:
            loss_cross_entropy = nn.CrossEntropyLoss().cuda()
        else:
            loss_cross_entropy = nn.CrossEntropyLoss()

        base_pred = (1. - self.mask) * y_pred + self.eps
        base_true = y_true
        base_cost = loss_cross_entropy(base_pred, base_true)

        abs_pred = torch.sum(mask * y_pred, -1)
        # add some small value to prevent NaN when prediction is abstained
        abs_pred = torch.clamp(abs_pred, self.eps, 1. - self.eps)

        return ((1. - abs_pred) * base_cost - alpha * torch.log(1. - abs_pred))
