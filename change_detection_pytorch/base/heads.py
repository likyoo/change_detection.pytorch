import torch.nn as nn
from .modules import Flatten, Activation, OCR


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1, align_corners=True):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.Upsample(scale_factor=upsampling, mode='bilinear', align_corners=align_corners) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class SegmentationOCRHead(nn.Module):

    def __init__(self, in_channels, out_channels, activation=None, upsampling=1, align_corners=True):
        super().__init__()
        self.ocr_head = OCR(in_channels, out_channels)
        self.upsampling = nn.Upsample(scale_factor=upsampling, mode='bilinear', align_corners=align_corners) if upsampling > 1 else nn.Identity()
        self.activation = Activation(activation)

    def forward(self, x):
        coarse_pre, pre = self.ocr_head(x)
        coarse_pre = self.activation(self.upsampling(coarse_pre))
        pre = self.activation(self.upsampling(pre))
        return [coarse_pre, pre]


class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2, activation=None):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        flatten = Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, dropout, linear, activation)
