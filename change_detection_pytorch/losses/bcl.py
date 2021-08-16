import torch
from torch.nn.modules.loss import _Loss


class BCLLoss(_Loss):
    """loss function of stanet"""
    def __init__(
            self,
            label_value: int = 1,
            margin: int = 2
    ):
        super(BCLLoss, self).__init__()
        self.margin = margin
        self.label_value = label_value

    def forward(self, distance, label):
        label = label.float()
        label[label == 255] = 1
        label[label == self.label_value] = -1
        label[label == 0] = 1
        mask = (label != 255).float()
        distance = distance * mask
        pos_num = torch.sum((label == 1).float()) + 0.0001
        neg_num = torch.sum((label == -1).float()) + 0.0001

        loss_1 = torch.sum((1 + label) / 2 * torch.pow(distance, 2)) / pos_num
        loss_2 = torch.sum((1 - label) / 2 * mask *
                           torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
                           ) / neg_num
        loss = loss_1 + loss_2
        return loss
