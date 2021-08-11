"""
The pipeline of Albumentations augmentation

"""

from __future__ import absolute_import

import warnings
from abc import ABC
from types import LambdaType

import numpy as np
import torch
from albumentations.core.transforms_interface import (BasicTransform,
                                                      DualTransform,
                                                      ImageOnlyTransform, NoOp,
                                                      to_tuple)
from albumentations.core.utils import format_args
from torchvision.transforms import functional as F

__all__ = ["ToTensorTest", "ChunkImage", "ExchangeTime"]


class ToTensorTest(BasicTransform):
    """Convert image and mask to `torch.Tensor`. The numpy `BHWC` image is converted to pytorch `BCHW` tensor.
    If the image is in `BHW` format (grayscale image), it will be converted to pytorch `BHW` tensor.
    Args:
        transpose_mask (bool): if True and an input mask has three dimensions, this transform will transpose dimensions
        so the shape `[height, width, num_channels]` becomes `[num_channels, height, width]`. The latter format is a
        standard format for PyTorch Tensors. Default: False.
    """

    def __init__(self, transpose_mask=False, always_apply=True, p=1.0):
        super(ToTensorTest, self).__init__(always_apply=always_apply, p=p)
        self.transpose_mask = transpose_mask

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask}

    def apply(self, img, **params):  # skipcq: PYL-W0613
        if len(img.shape) not in [3, 4]:
            raise ValueError("Albumentations only supports images in BHW or BHWC format")

        if len(img.shape) == 3:
            img = np.expand_dims(img, 4)

        return torch.from_numpy(img.transpose(0, 3, 1, 2))

    def apply_to_mask(self, mask, **params):  # skipcq: PYL-W0613
        if self.transpose_mask and mask.ndim == 4:
            mask = mask.transpose(0, 3, 1, 2)
        return torch.from_numpy(mask)

    def get_transform_init_args_names(self):
        return ("transpose_mask",)

    def get_params_dependent_on_targets(self, params):
        return {}


class ChunkImage(DualTransform):
    """Slice the image into uniform chunks.
    Args:
        p (float): probability of applying the transform. Default: 1.0
    Targets:
        image, mask
    Image types:
        uint8, float32
    """

    def __init__(
            self,
            size=256,
            always_apply=True,
            p=1,
    ):
        super(ChunkImage, self).__init__(always_apply, p)
        self.size = size

    def chunk(self, data, size):
        h, w = data.shape[:2]
        patch_num = h // size
        if data.ndim == 3:
            c = data.shape[-1]
            data = np.lib.stride_tricks.as_strided(data, (patch_num, patch_num, size, size, c),
                                                   tuple(
                                                       np.array([size * h * c, size * c, h * c, c, 1]) * data.itemsize))
            # data (4, 4, 256, 256, 3)
            data = np.reshape(data, (-1, size, size, c))
        elif data.ndim == 2:
            data = np.lib.stride_tricks.as_strided(data, (patch_num, patch_num, size, size),
                                                   tuple(np.array([size * h, size, h, 1]) * data.itemsize))
            # data (4, 4, 256, 256)
            data = np.reshape(data, (-1, size, size))
        else:
            raise ValueError('the {}-dim data is not supported'.format(data.ndim))

        return data

    def apply(self, img, **params):
        return self.chunk(img, self.size)

    def apply_to_mask(self, img, **params):
        return self.chunk(img, self.size)

    def get_transform_init_args_names(self):
        return (
            "size",
        )


class ExchangeTime(ImageOnlyTransform):
    """Exchange images of different times.
    Args:
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(
            self,
            always_apply=False,
            p=0.5,
    ):
        super(ExchangeTime, self).__init__(always_apply, p)

    def apply(self, img, **params):
        if not (img.shape[-1] & 1) == 0:
            raise ValueError("ExchangeTime: Unequal number of channels")
        idx1 = img.shape[-1] // 2
        idx2 = img.shape[-1] + 1
        return np.dstack((img[:, :, idx1:idx2], img[:, :, 0:idx1]))
