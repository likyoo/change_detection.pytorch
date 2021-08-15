from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

__all__ = ["HybridLoss"]


# Custom combination of multiple loss functions
# For example:
# loss1 = cdp.utils.losses.CrossEntropyLoss()
# loss2 = cdp.losses.DiceLoss(mode='multiclass')
# loss = cdp.losses.HybridLoss(loss1, loss2, reduction='sum')

class HybridLoss(_Loss):
    __name__ = "HybridLoss"

    def __init__(
            self,
            loss1: _Loss,
            loss2: _Loss,
            reduction: Optional[str] = "mean",
    ):
        """Implementation of Hybrid loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            loss1: The first loss function.
            loss2: The second loss function.
            reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Default: ``'mean'``

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        """
        super(HybridLoss, self).__init__()

        self.loss1 = loss1
        self.loss2 = loss2
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.size(0) == y_pred.size(0)

        loss1 = self.loss1(y_pred, y_true)
        loss2 = self.loss2(y_pred, y_true)
        loss = torch.stack([loss1, loss2], dim=0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass
        else:
            raise ValueError('reduction="{}" is not defined'.format(self.reduction))

        return loss
