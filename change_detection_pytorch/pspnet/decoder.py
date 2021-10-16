import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import Decoder, modules


class PSPBlock(nn.Module):

    def __init__(self, in_channels, out_channels, pool_size, use_bathcnorm=True):
        super().__init__()
        if pool_size == 1:
            use_bathcnorm = False  # PyTorch does not support BatchNorm for 1x1 shape
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
            modules.Conv2dReLU(in_channels, out_channels, (1, 1), use_batchnorm=use_bathcnorm)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = self.pool(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x


class PSPModule(nn.Module):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6), use_bathcnorm=True):
        super().__init__()

        self.blocks = nn.ModuleList([
            PSPBlock(in_channels, in_channels // len(sizes), size, use_bathcnorm=use_bathcnorm) for size in sizes
        ])

    def forward(self, x):
        xs = [block(x) for block in self.blocks] + [x]
        x = torch.cat(xs, dim=1)
        return x


class PSPDecoder(Decoder):

    def __init__(
            self,
            encoder_channels,
            use_batchnorm=True,
            out_channels=512,
            dropout=0.2,
            fusion_form="concat",
    ):
        super().__init__()

        # adjust encoder channels according to fusion form
        self.fusion_form = fusion_form
        if self.fusion_form in self.FUSION_DIC["2to2_fusion"]:
            encoder_channels = [ch*2 for ch in encoder_channels]

        self.psp = PSPModule(
            in_channels=encoder_channels[-1],
            sizes=(1, 2, 3, 6),
            use_bathcnorm=use_batchnorm,
        )

        self.conv = modules.Conv2dReLU(
            in_channels=encoder_channels[-1] * 2,
            out_channels=out_channels,
            kernel_size=1,
            use_batchnorm=use_batchnorm,
        )

        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, *features):
        # features = self.aggregation_layer(features[0], features[1],
        #                                   self.fusion_form, ignore_original_img=True)
        # x = features[-1]
        x = self.fusion(features[0][-1], features[1][-1], self.fusion_form)
        x = self.psp(x)
        x = self.conv(x)
        x = self.dropout(x)

        return x
