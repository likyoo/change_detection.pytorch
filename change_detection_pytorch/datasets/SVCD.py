import os.path as osp

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from change_detection_pytorch.datasets.custom import CustomDataset


class SVCD_Dataset(CustomDataset):
    """ season-varying change detection dataset"""

    def __init__(self, img_dir, sub_dir_1='A', sub_dir_2='B', ann_dir=None, img_suffix='.jpg', seg_map_suffix='.jpg',
                 transform=None, split=None, data_root=None, test_mode=False, size=256, debug=False):
        super().__init__(img_dir, sub_dir_1, sub_dir_2, ann_dir, img_suffix, seg_map_suffix, transform, split,
                         data_root, test_mode, size, debug)

    def get_default_transform(self):
        """Set the default transformation."""

        default_transform = A.Compose([
            A.Resize(self.size, self.size),
            # A.HorizontalFlip(p=0.5),
            # A.RandomRotate90(p=0.5),
            A.Normalize(mean=(0, 0, 0, 0, 0, 0), std=(1, 1, 1, 1, 1, 1)),     # div(255)
            ToTensorV2()
        ])
        return default_transform

    def get_test_transform(self):
        """Set the test transformation."""

        test_transform = A.Compose([
            A.Resize(self.size, self.size),
            A.Normalize(mean=(0, 0, 0, 0, 0, 0), std=(1, 1, 1, 1, 1, 1)),     # div(255)
            ToTensorV2()
        ])
        return test_transform

    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data (with annotation if ann_dir is not None).
        """

        to_tensor_axis_bias = 2 if self.debug else 0

        if not self.ann_dir:
            ann = None
            img1, img2, filename = self.prepare_img(idx)
            img = np.concatenate((img1, img2), axis=2)
            transformed_image = self.transform(image=img)['image']
        else:
            img1, img2, ann, filename = self.prepare_img_ann(idx)
            img = np.concatenate((img1, img2), axis=2)
            transformed_data = self.transform(image=img, mask=ann)
            transformed_image, ann = transformed_data['image'], transformed_data['mask']

        img1, img2 = np.split(transformed_image, 2, axis=0+to_tensor_axis_bias)

        return img1, img2, ann, filename
