import torch
from typing import Optional
from torch.nn import functional as F
from ..encoders import get_encoder
from .decoder import STANetDecoder


class STANet(torch.nn.Module):
    def __init__(
            self,
            encoder_name: str = "resnet",
            encoder_weights: Optional[str] = "imagenet",
            sa_mode: str = "PAM",
            in_channels: int = 3,
    ):
        super(STANet, self).__init__()
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            weights=encoder_weights
        )

        self.decoder = STANetDecoder(
            encoder_out_channels=self.encoder.out_channels,
            sa_mode=sa_mode
        )

    def forward(self, x1, x2):
        # only support siam encoder
        features = self.encoder(x1), self.encoder(x2)
        features = self.decoder(*features)
        dist = F.pairwise_distance(features[0], features[1],keepdim=True)
        dist = F.interpolate(dist, x1.shape[2:], mode='bilinear', align_corners=True)
        return dist
