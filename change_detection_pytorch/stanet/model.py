import torch
from typing import Optional
from torch.nn import functional as F
from ..encoders import get_encoder
from .decoder import STANetDecoder
from ..base import SegmentationHead


class STANet(torch.nn.Module):
    """
    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
        return_distance_map: If True, return distance map, which shape is (BatchSize, Height, Width), of feature maps from images of two periods. Default False.

    Returns:
        ``torch.nn.Module``: STANet

    .. STANet:
        https://www.mdpi.com/2072-4292/12/10/1662

    """

    def __init__(
            self,
            encoder_name: str = "resnet",
            encoder_weights: Optional[str] = "imagenet",
            sa_mode: str = "PAM",
            in_channels: int = 3,
            classes=2,
            activation=None,
            return_distance_map=False,
            **kwargs
    ):
        super(STANet, self).__init__()
        self.return_distance_map = return_distance_map
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            weights=encoder_weights
        )

        self.decoder = STANetDecoder(
            encoder_out_channels=self.encoder.out_channels,
            sa_mode=sa_mode
        )
        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channel * 2,
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

    def forward(self, x1, x2):
        # only support siam encoder
        features = self.encoder(x1), self.encoder(x2)
        features = self.decoder(*features)
        if self.return_distance_map:
            dist = F.pairwise_distance(features[0], features[1], keepdim=True)
            dist = F.interpolate(dist, x1.shape[2:], mode='bilinear', align_corners=True)
            return dist
        else:
            decoder_output = torch.cat([features[0], features[1]], dim=1)
            decoder_output = F.interpolate(decoder_output, x1.shape[2:], mode='bilinear', align_corners=True)
            masks = self.segmentation_head(decoder_output)
            return masks
