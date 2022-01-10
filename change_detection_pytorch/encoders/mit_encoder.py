# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from pretrainedmodels.models.torchvision_models import pretrained_settings

from ._base import EncoderMixin
from .mix_transformer import MixVisionTransformer


class MixVisionTransformerEncoder(MixVisionTransformer, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

    def get_stages(self):
        return [nn.Identity()]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for stage in stages:
            x = stage(x)
            features.append(x)
        outs = self.forward_features(x)
        add_feature = F.interpolate(outs[0], scale_factor=2)
        features = features + [add_feature] + outs
        return features

    def load_state_dict(self, state_dict, **kwargs):
        new_state_dict = {}
        if state_dict.get('state_dict'):
            state_dict = state_dict['state_dict']
            for k, v in state_dict.items():
                if k.startswith('backbone'):
                    new_state_dict[k.replace('backbone.', '')] = v
        else:
            new_state_dict = deepcopy(state_dict)
        super().load_state_dict(new_state_dict, **kwargs)


# https://github.com/NVlabs/SegFormer
new_settings = {
    "mit-b0": {
        "imagenet": "https://lino.local.server/mit_b0.pth"
    },
    "mit-b1": {
        "imagenet": "https://lino.local.server/mit_b1.pth"
    },
    "mit-b2": {
        "imagenet": "https://lino.local.server/mit_b2.pth"
    },
    "mit-b3": {
        "imagenet": "https://lino.local.server/mit_b3.pth"
    },
    "mit-b4": {
        "imagenet": "https://lino.local.server/mit_b4.pth"
    },
    "mit-b5": {
        "imagenet": "https://lino.local.server/mit_b5.pth"
    },
}

pretrained_settings = deepcopy(pretrained_settings)
for model_name, sources in new_settings.items():
    if model_name not in pretrained_settings:
        pretrained_settings[model_name] = {}

    for source_name, source_url in sources.items():
        pretrained_settings[model_name][source_name] = {
            "url": source_url,
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }

mit_encoders = {
    "mit-b0": {
        "encoder": MixVisionTransformerEncoder,
        "pretrained_settings": pretrained_settings["mit-b0"],
        "params": {
            "patch_size": 4,
            "embed_dims": [32, 64, 160, 256],
            "num_heads": [1, 2, 5, 8],
            "mlp_ratios": [4, 4, 4, 4],
            "qkv_bias": True,
            "norm_layer": partial(nn.LayerNorm, eps=1e-6),
            "depths": [2, 2, 2, 2],
            "sr_ratios": [8, 4, 2, 1],
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
            "out_channels": (3, 32, 32, 64, 160, 256)
        }
    },
    "mit-b1": {
        "encoder": MixVisionTransformerEncoder,
        "pretrained_settings": pretrained_settings["mit-b1"],
        "params": {
            "patch_size": 4,
            "embed_dims": [64, 128, 320, 512],
            "num_heads": [1, 2, 5, 8],
            "mlp_ratios": [4, 4, 4, 4],
            "qkv_bias": True,
            "norm_layer": partial(nn.LayerNorm, eps=1e-6),
            "depths": [2, 2, 2, 2],
            "sr_ratios": [8, 4, 2, 1],
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
            "out_channels": (3, 64, 64, 128, 320, 512)
        }
    },
    "mit-b2": {
        "encoder": MixVisionTransformerEncoder,
        "pretrained_settings": pretrained_settings["mit-b2"],
        "params": {
            "patch_size": 4,
            "embed_dims": [64, 128, 320, 512],
            "num_heads": [1, 2, 5, 8],
            "mlp_ratios": [4, 4, 4, 4],
            "qkv_bias": True,
            "norm_layer": partial(nn.LayerNorm, eps=1e-6),
            "depths": [3, 4, 6, 3],
            "sr_ratios": [8, 4, 2, 1],
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
            "out_channels": (3, 64, 64, 128, 320, 512)
        }
    },
    "mit-b3": {
        "encoder": MixVisionTransformerEncoder,
        "pretrained_settings": pretrained_settings["mit-b3"],
        "params": {
            "patch_size": 4,
            "embed_dims": [64, 128, 320, 512],
            "num_heads": [1, 2, 5, 8],
            "mlp_ratios": [4, 4, 4, 4],
            "qkv_bias": True,
            "norm_layer": partial(nn.LayerNorm, eps=1e-6),
            "depths": [3, 4, 18, 3],
            "sr_ratios": [8, 4, 2, 1],
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
            "out_channels": (3, 64, 64, 128, 320, 512)
        }
    },
    "mit-b4": {
        "encoder": MixVisionTransformerEncoder,
        "pretrained_settings": pretrained_settings["mit-b4"],
        "params": {
            "patch_size": 4,
            "embed_dims": [64, 128, 320, 512],
            "num_heads": [1, 2, 5, 8],
            "mlp_ratios": [4, 4, 4, 4],
            "qkv_bias": True,
            "norm_layer": partial(nn.LayerNorm, eps=1e-6),
            "depths": [3, 8, 27, 3],
            "sr_ratios": [8, 4, 2, 1],
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
            "out_channels": (3, 64, 64, 128, 320, 512)
        }
    },
    "mit-b5": {
        "encoder": MixVisionTransformerEncoder,
        "pretrained_settings": pretrained_settings["mit-b5"],
        "params": {
            "patch_size": 4,
            "embed_dims": [64, 128, 320, 512],
            "num_heads": [1, 2, 5, 8],
            "mlp_ratios": [4, 4, 4, 4],
            "qkv_bias": True,
            "norm_layer": partial(nn.LayerNorm, eps=1e-6),
            "depths": [3, 6, 40, 3],
            "sr_ratios": [8, 4, 2, 1],
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
            "out_channels": (3, 64, 64, 128, 320, 512)
        }
    },
}
