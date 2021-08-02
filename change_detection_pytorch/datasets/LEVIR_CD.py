from change_detection_pytorch.datasets.custom import CustomDataset
import os.path as osp
import numpy as np


class LEVIR_CD_Dataset(CustomDataset):
    """LEVIR-CD dataset"""

    def __init__(self, img_dir, sub_dir_1='A', sub_dir_2='B', img_suffix='.png', ann_dir=None, seg_map_suffix='.png',
                 transform=None, split=None, data_root=None, test_mode=False, size=256, debug=False):
        super().__init__(img_dir, sub_dir_1, sub_dir_2, img_suffix, ann_dir, seg_map_suffix, transform, split,
                         data_root, test_mode, size, debug)

    def get_default_transform(self):
        """Set the default transformation."""

        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        default_transform = A.Compose([
            A.RandomCrop(256, 256),
            # A.ShiftScaleRotate(),
            A.Normalize(mean=(0, 0, 0, 0, 0, 0), std=(1, 1, 1, 1, 1, 1)),  # div(255)
            ToTensorV2()
        ])
        return default_transform

    def get_test_transform(self):
        """Set the test transformation."""

        import albumentations as A
        from change_detection_pytorch.datasets.transforms.albu import ChunkImage, ToTensorTest
        test_transform = A.Compose([
            A.Normalize(mean=(0, 0, 0, 0, 0, 0), std=(1, 1, 1, 1, 1, 1)),  # div(255)
            ChunkImage(self.size),
            ToTensorTest(),
        ])
        return test_transform

    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        to_tensor_axis_bias = 2 if self.debug else 0

        if self.test_mode:
            img_info = self.prepare_train_img(idx)
            img = np.concatenate((img_info['img']['img1'], img_info['img']['img2']), axis=2)

            test_transform = self.get_test_transform()
            transformed_data = test_transform(image=img, mask=img_info['ann']['ann'])
            transformed_image, img_info['ann']['ann'] = transformed_data['image'], transformed_data['mask']
            img_info['img']['img1'], img_info['img']['img2'] = np.split(transformed_image, 2, axis=1+to_tensor_axis_bias)

        else:
            img_info = self.prepare_train_img(idx)
            img = np.concatenate((img_info['img']['img1'], img_info['img']['img2']), axis=2)
            transformed_data = self.transform(image=img, mask=img_info['ann']['ann'])
            transformed_image, img_info['ann']['ann'] = transformed_data['image'], transformed_data['mask']
            img_info['img']['img1'], img_info['img']['img2'] = np.split(transformed_image, 2, axis=0+to_tensor_axis_bias)

        return img_info

if __name__ == "__main__":
    LEVIR_CD_Dataset('dir')
