"""
The pipeline of Albumentations augmentation.

"""

from __future__ import absolute_import

import random
import warnings
from abc import ABC
from collections.abc import Sequence
from types import LambdaType

import numpy as np
import torch
from albumentations.core.transforms_interface import (BasicTransform,
                                                      DualTransform,
                                                      ImageOnlyTransform, NoOp,
                                                      to_tuple)
from albumentations.core.utils import format_args
from torchvision.transforms import functional as F
import cv2

try:
    from albumentations.augmentations.functional import random_crop
except:
    from albumentations.augmentations.crops.functional import random_crop

__all__ = ["ToTensorTest", "ChunkImage", "ExchangeTime", "RandomChoice", "Mosaic"]


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
            # data (1024, 1024, 3)
            c = data.shape[-1]
            data = np.lib.stride_tricks.as_strided(data, (patch_num, patch_num, size, size, c),
                                                   tuple(
                                                       np.array([size * h * c, size * c, h * c, c, 1]) * data.itemsize))
            # data (4, 4, 256, 256, 3)
            data = np.reshape(data, (-1, size, size, c))
            # data (16, 256, 256, 3)
        elif data.ndim == 2:
            data = np.lib.stride_tricks.as_strided(data, (patch_num, patch_num, size, size),
                                                   tuple(np.array([size * h, size, h, 1]) * data.itemsize))
            # data (4, 4, 256, 256)
            data = np.reshape(data, (-1, size, size))
            # data (16, 256, 256)
        else:
            raise ValueError('the {}-dim data is not supported'.format(data.ndim))

        return data

    def apply(self, img, **params):
        return self.chunk(img, self.size)

    def apply_to_mask(self, mask, **params):
        return self.chunk(mask, self.size)

    def get_transform_init_args_names(self):
        return (
            "size",
        )


class ExchangeTime(BasicTransform):
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

    def __call__(self, force_apply=False, **kwargs):
        if self.replay_mode:
            if self.applied_in_replay:
                return self.apply_with_params(self.params, **kwargs)

            return kwargs

        if (random.random() < self.p) or self.always_apply or force_apply:
            kwargs['image'], kwargs['image_2'] = kwargs['image_2'], kwargs['image']

        return kwargs


class RandomChoice(BasicTransform):
    """Apply single transformation randomly picked from a list.
    """

    def __init__(self, transforms, always_apply=True, p=1.0):
        super(RandomChoice, self).__init__(always_apply=always_apply, p=p)
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence")
        self.transforms = transforms

    def __call__(self, force_apply=False, **kwargs):
        t = random.choice(self.transforms)
        return t(force_apply=force_apply, **kwargs)


class Mosaic(BasicTransform):
    """ Mosaic?
    Args:
        size (int): input_size / 2
        img_infos (dict): a dict with img info.
        dataset_size (int): The size (len) of dataset.
        p (float): probability of applying the transform. Default: 0.5
    Targets:
        image, mask
    Image types:
        uint8, float32
    """

    def __init__(
            self,
            size=256,
            img_infos=None,
            dataset_size=None,
            always_apply=False,
            p=0.5,
    ):
        super(Mosaic, self).__init__(always_apply, p)
        self.size = size
        self.img_infos = img_infos
        self.dataset_size = dataset_size

    def __call__(self, force_apply=False, **kwargs):
        if self.replay_mode:
            if self.applied_in_replay:
                return self.apply_with_params(self.params, **kwargs)

            return kwargs

        if (random.random() < self.p) or self.always_apply or force_apply:
            params = self.get_params()
            for key, arg in kwargs.items():
                kwargs[key] = self.apply(arg, key, **params)

        return kwargs

    def get_image(self, img_info, img_key="image", vis=False):
        """Open and read the image.
        Args:
            img_info (dict): a dict with img info.
            img_key (str):
            vis (bool): Whether to use visualization (debug mode).
        Returns:
            dict: image info with new keys.
        """

        if img_key == "image":
            img = cv2.cvtColor(cv2.imread(img_info['img']['img1_path']), cv2.COLOR_BGR2RGB)
        elif img_key == "image_2":
            img = cv2.cvtColor(cv2.imread(img_info['img']['img2_path']), cv2.COLOR_BGR2RGB)
        elif img_key == "mask":
            img = cv2.imread(img_info['ann']['ann_path'], cv2.IMREAD_GRAYSCALE)
            img = img / 255 if not vis else img
        else:
            raise ValueError("Undefined key: {}".format(img_key))
        return img

    def apply(self, img, img_key="image", **params):

        img1 = random_crop(img, self.size, self.size, params["h_start_1"], params["w_start_1"])

        img_info_2 = self.img_infos[params["index_2"]]
        img2 = self.get_image(img_info_2, img_key)
        img2 = random_crop(img2, self.size, self.size, params["h_start_2"], params["w_start_2"])

        img_info_3 = self.img_infos[params["index_3"]]
        img3 = self.get_image(img_info_3, img_key)
        img3 = random_crop(img3, self.size, self.size, params["h_start_3"], params["w_start_3"])

        img_info_4 = self.img_infos[params["index_4"]]
        img4 = self.get_image(img_info_4, img_key)
        img4 = random_crop(img4, self.size, self.size, params["h_start_4"], params["w_start_4"])

        img = np.concatenate([np.concatenate([img1, img2], axis=1),
                              np.concatenate([img3, img4], axis=1)], axis=0)

        return img

    def get_params(self):
        return {"h_start_1": random.random(), "w_start_1": random.random(),
                "h_start_2": random.random(), "w_start_2": random.random(),
                "h_start_3": random.random(), "w_start_3": random.random(),
                "h_start_4": random.random(), "w_start_4": random.random(),
                "index_2": random.randint(0, self.dataset_size-1),
                "index_3": random.randint(0, self.dataset_size-1),
                "index_4": random.randint(0, self.dataset_size-1),}

    def get_transform_init_args_names(self):
        return (
            "size",
            "img_infos",
            "dataset_size",
            )

