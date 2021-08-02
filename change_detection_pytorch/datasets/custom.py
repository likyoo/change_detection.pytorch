import os
import os.path as osp
from collections import OrderedDict
from functools import reduce
import numpy as np
from torch.utils.data import Dataset
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CustomDataset(Dataset):
    """Custom datasets for change detection. An example of file structure
    is as followed.
    .. code-block:: none
        ├── data
        │   ├── my_dataset
        │   │   ├── train
        │   │   │   ├── img1_dir
        │   │   │   ├── img1_dir
        │   │   │   ├── label_dir
        │   │   ├── val
        │   │   │   ├── img1_dir
        │   │   │   ├── img1_dir
        │   │   │   ├── label_dir

    The imgs/gt pair of CustomDataset should be of the same except suffix.
    A valid imgs/gt filename pair should be like ``xxx{img_suffix}`` and
    ``xxx{seg_map_suffix}`` (extension is also included in the suffix).
    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): Whether to the test mode.
        sub_dir_1 (str): Path to the directory of the first temporal images.
            e.g. 'A' in LEVIR-CD dataset (LEVIR-CD/train/A). Default: 'A'
        sub_dir_2 (str): Path to the directory of the second temporal images.
            e.g. 'B' in LEVIR-CD dataset (LEVIR-CD/train/B). Default: 'B'
        size (int): The size of input images.
        debug (bool): Whether to use debug mode. i.e. visualization.
    """

    def __init__(self,
                 img_dir,
                 sub_dir_1='A',
                 sub_dir_2='B',
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 transform=None,
                 split=None,
                 data_root=None,
                 test_mode=False,
                 size=256,
                 debug=False):
        self.transform = transform
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.sub_dir_1 = sub_dir_1
        self.sub_dir_2 = sub_dir_2
        self.size = size
        self.debug = debug

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)

        # transform/augment data
        if transform is None:
            self.transform = self.get_default_transform()

        # debug, visualize augmentations
        if self.debug:
            self.transform = A.Compose([t for t in self.transform if not isinstance(t, (A.Normalize, ToTensorV2))])

        # load annotations
        self.img_infos = self.load_infos(self.img_dir, self.img_suffix,
                                         self.seg_map_suffix, self.sub_dir_1,
                                         self.sub_dir_2, self.ann_dir,
                                         self.split)

    def load_infos(self, img_dir, img_suffix, seg_map_suffix, sub_dir_1,
                         sub_dir_2, ann_dir, split):
        """Load annotation from directory.
        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None
        Returns:
            list[dict]: All image info of datasets.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name)
                    img_info['img'] = dict(img1_path=osp.join(img_dir, sub_dir_1, img_name),
                                           img2_path=osp.join(img_dir, sub_dir_2, img_name))
                    if ann_dir is not None:
                        seg_map = osp.join(ann_dir,
                                           img_name.replace(img_suffix, seg_map_suffix))
                        img_info['ann'] = dict(ann_path=seg_map)
                    img_infos.append(img_info)
        else:
            for img in glob.glob(osp.join(img_dir, sub_dir_1, '*'+img_suffix)):
                img_name = osp.basename(img)
                img_info = dict(filename=img_name)
                img_info['img'] = dict(img1_path=osp.join(img_dir, sub_dir_1, img_name),
                                       img2_path=osp.join(img_dir, sub_dir_2, img_name))
                if ann_dir is not None:
                    seg_map = osp.join(ann_dir,
                                       img_name.replace(img_suffix, seg_map_suffix))
                    img_info['ann'] = dict(ann_path=seg_map)
                img_infos.append(img_info)

        print(f'Loaded {len(img_infos)} images')
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def get_default_transform(self):
        """Set the default transformation."""

        default_transform = A.Compose([
            A.Resize(self.size, self.size),
            A.Normalize(mean=(0, 0, 0, 0, 0, 0), std=(1, 1, 1, 1, 1, 1)),     # div(255)
            ToTensorV2()
        ])
        return default_transform

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        self.get_image(img_info)
        self.get_gt_seg_maps(img_info, self.debug)
        return img_info

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        self.get_image(img_info)
        return img_info

    def get_image(self, img_info):
        """Open and read the image.
        Args:
            img_info (dict): a dict with img info.
        Returns:
            dict: image info with new keys.
        """

        import cv2
        img_info['img']['img1'] = cv2.cvtColor(cv2.imread(img_info['img']['img1_path']),
                                               cv2.COLOR_BGR2RGB)
        img_info['img']['img2'] = cv2.cvtColor(cv2.imread(img_info['img']['img2_path']),
                                               cv2.COLOR_BGR2RGB)
        return img_info

    def get_gt_seg_maps(self, img_info, vis=False):
        """Open and read the ground truth.
        Args:
            img_info (dict): a dict with ann info.
            vis (bool): Whether to use visualization (debug mode).
        Returns:
            dict: ann info with new keys.
        """

        import cv2
        ann = cv2.imread(img_info['ann']['ann_path'], cv2.IMREAD_GRAYSCALE)
        if not vis:
            ann = ann / 255
        img_info['ann']['ann'] = ann
        return img_info

    def format_results(self, results, **kwargs):
        """Place holder to format result to datasets specific output."""

        pass

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
            img_info = self.prepare_test_img(idx)
            img = np.concatenate((img_info['img']['img1'], img_info['img']['img2']), axis=2)
            transformed_image = self.transform(image=img)['image']

        else:
            img_info = self.prepare_train_img(idx)
            img = np.concatenate((img_info['img']['img1'], img_info['img']['img2']), axis=2)
            transformed_data = self.transform(image=img, mask=img_info['ann']['ann'])
            transformed_image, img_info['ann']['ann'] = transformed_data['image'], transformed_data['mask']

        img_info['img']['img1'], img_info['img']['img2'] = np.split(transformed_image, 2, axis=0+to_tensor_axis_bias)
        return img_info

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)