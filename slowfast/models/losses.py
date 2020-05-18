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
        n_cl_preds = output[:, 0]
        len_preds = output[:,0: 6]
        class_preds = output[:,6:].reshape(-1, 5, 5)

        n_classes_loss = nn.SmoothL1Loss(reduction='mean')
        len_loss = nn.SmoothL1Loss(reduction='mean')
        class_loss = nn.CrossEntropyLoss(reduction='mean')

        vn_cl_loss = n_classes_loss(n_cl_preds, target[:, 0])
        vlen_loss = len_loss(len_preds, target[:, 0: 6])
        vcl_loss = class_loss(class_preds, target[:, 6:].long()).float()
        resulted_loss = vn_cl_loss + vlen_loss + 2 * vcl_loss
        return resulted_loss
_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "regression": RegressionLoss
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
