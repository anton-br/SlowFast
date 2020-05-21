#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss


class RegressionLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(RegressionLoss, self).__init__()
        self.reduction = reduction

    def forward(self, output, target):
        output = output.float()
        target = target.float()
        ln = (target.size(1) - 1) // 2 + 1
        len_preds = output[:,: ln]
        class_preds = output[:,ln:].reshape(target.size(0), ln-1, 5)
        class_preds = class_preds.transpose(1, 2)

        len_loss = nn.MSELoss(reduction='mean')
        class_loss = nn.CrossEntropyLoss(reduction='mean')

        vlen_loss = len_loss(len_preds, target[:,: ln])
        vcl_loss = class_loss(class_preds, target[:,ln:].long()).float()
        resulted_loss = vlen_loss + 10 * vcl_loss
        return resulted_loss

class StendLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(StendLoss, self).__init__()
        self.reduction = reduction

    def forward(self, output, target):
        start_pred = output[:,0]
        end_pred = output[:,1]

        start_target = target[0]
        end_target = target[1]

        start_loss = nn.BCEWithLogitsLoss(reduction=self.reduction)
        end_loss = nn.BCEWithLogitsLoss(reduction=self.reduction)

        start_comp = start_loss(start_pred, start_target)
        end_comp = end_loss(end_pred, end_target)
        return start_comp + end_comp

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "regression": RegressionLoss,
    'mse': nn.MSELoss,
    'stend': StendLoss,
}

def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
