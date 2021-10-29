"""
BSD 2-Clause License

Copyright (c) 2020, justchenhao
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAG
"""
import torch
import torch.nn.functional as F
from torch import nn
from ..base import Decoder
from .BAM import BAM
from .PAM2 import PAM


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class STANetDecoder(Decoder):
    def __init__(
            self,
            encoder_out_channels,
            f_c=64,
            sa_mode='PAM'
    ):
        super(STANetDecoder, self).__init__()
        self.out_channel = f_c
        self.backbone_decoder = BackboneDecoder(f_c, nn.BatchNorm2d, encoder_out_channels)
        self.netA = CDSA(in_c=f_c, ds=1, mode=sa_mode)

    def forward(self, *features):
        # fetch feature maps
        feature_0 = features[0]
        feature_1 = features[1]
        # 1x1 conv and concatenation feature maps
        feature_0 = self.backbone_decoder(feature_0[5], feature_0[2], feature_0[3], feature_0[4])
        feature_1 = self.backbone_decoder(feature_1[5], feature_1[2], feature_1[3], feature_1[4])
        feature_0, feature_1 = self.netA(feature_0, feature_1)
        return feature_0, feature_1


class CDSA(nn.Module):
    """self attention module for change detection

    """

    def __init__(self, in_c, ds=1, mode='BAM'):
        super(CDSA, self).__init__()
        self.in_C = in_c
        self.ds = ds
        # print('ds: ', self.ds)
        self.mode = mode
        if self.mode == 'BAM':
            self.Self_Att = BAM(self.in_C, ds=self.ds)
        elif self.mode == 'PAM':
            self.Self_Att = PAM(in_channels=self.in_C, out_channels=self.in_C, sizes=[1, 2, 4, 8], ds=self.ds)
        self.apply(weights_init)

    def forward(self, x1, x2):
        height = x1.shape[3]
        x = torch.cat((x1, x2), 3)
        x = self.Self_Att(x)
        return x[:, :, :, 0:height], x[:, :, :, height:]


class BackboneDecoder(nn.Module):
    def __init__(self, fc, BatchNorm, encoder_out_channels):
        super(BackboneDecoder, self).__init__()
        self.fc = fc
        self.dr2 = DR(encoder_out_channels[2], 96)
        self.dr3 = DR(encoder_out_channels[3], 96)
        self.dr4 = DR(encoder_out_channels[4], 96)
        self.dr5 = DR(encoder_out_channels[5], 96)
        self.last_conv = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, self.fc, kernel_size=1, stride=1, padding=0, bias=False),
                                       BatchNorm(self.fc),
                                       nn.ReLU(),
                                       )

        self._init_weight()

    def forward(self, x, low_level_feat2, low_level_feat3, low_level_feat4):

        # x1 = self.dr1(low_level_feat1)
        x2 = self.dr2(low_level_feat2)
        x3 = self.dr3(low_level_feat3)
        x4 = self.dr4(low_level_feat4)
        x = self.dr5(x)
        x = F.interpolate(x, size=x2.size()[2:], mode='bilinear', align_corners=True)
        # x2 = F.interpolate(x2, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=x2.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x, x2, x3, x4), dim=1)

        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DR(nn.Module):
    def __init__(self, in_d, out_d):
        super(DR, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv1 = nn.Conv2d(self.in_d, self.out_d, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_d)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        return x

# if __name__ == '__main__':
#     from change_detection_pytorch.encoders import get_encoder
#
#     samples = torch.ones([1, 3, 256, 256])
#     encoder = get_encoder('resnet34')
#     features0 = encoder(samples)
#     for fc in features0:
#         print(fc.size())
#     features1 = encoder(samples)
#     model = STANetDecoder(encoder_out_channels=encoder.out_channels)
#     features0, features1 = model(features0, features1)
#     print(features0, features1)
