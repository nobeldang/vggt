import os.path as osp
import numpy as np

from .base_stereo_view_dataset import BaseStereoViewDataset
import h5py
import os
import torch
import numpy as np
import PIL.Image
from PIL.ImageOps import exif_transpose
import torchvision.transforms as tvf
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa
import math


DATA_ROOT='./data/megadepth1500' 


def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """ Open an image or a depthmap with opencv-python.
    """
    if path.endswith(('.exr', 'EXR')):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f'Could not load image={path} with {options=}')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

class MegaDepth_valid(BaseStereoViewDataset):
    def __init__(self, *args, **kwargs):
        self.ROOT = DATA_ROOT
        super().__init__(*args, **kwargs)
        self.metadata = dict(np.load(osp.join(self.ROOT, f'megadepth_meta_test.npz'), allow_pickle=True))
        with open(osp.join(self.ROOT, f'megadepth_test_pairs.txt'), 'r') as f:
            self.scenes = f.readlines()
        self.load_depth = False

    def __len__(self):
        return len(self.scenes)

    def _get_single_view(self, view_idx, resolution):
        """
        Load and preprocess a single image using MegaDepth's metadata and cropping logic.
        """
        import os.path as osp

        input_image_filename = osp.join(self.ROOT, view_idx)
        input_rgb_image = imread_cv2(input_image_filename)

        intrinsics = np.float32(self.metadata[view_idx].item()['intrinsic'])
        camera_pose = np.linalg.inv(np.float32(self.metadata[view_idx].item()['pose']))  # cam2world

        image, intrinsics = self._crop_resize_if_necessary(
            input_rgb_image, intrinsics, resolution = resolution, rng=None, info=(self.ROOT, view_idx))
        
        image = self.transform(image)

        image = (image + 1)/2   # to be between 0 and 1 for VGGT input
        return dict(
            img=image,
            camera_pose=camera_pose,  # cam2world
            camera_intrinsics=intrinsics,
            dataset='MegaDepth',
            label=self.ROOT,
            instance=view_idx
        )


    def _get_views(self, idx, resolution,  rng):
        """
        load data for megadepth_validation views
        """
        # load metadata
        views = []
        image_idx1, image_idx2 = self.scenes[idx].strip().split(' ')
        view_idxs = [image_idx1, image_idx2]
        for view_idx in view_idxs:
            input_image_filename = osp.join(self.ROOT, view_idx)
            # load rgb images
            input_rgb_image = imread_cv2(input_image_filename)
            # load metadata
            intrinsics = np.float32(self.metadata[view_idx].item()['intrinsic'])
            camera_pose = np.linalg.inv(np.float32(self.metadata[view_idx].item()['pose']))

            image, intrinsics = self._crop_resize_if_necessary(
                input_rgb_image, intrinsics, resolution, rng=rng, info=(self.ROOT, view_idx))
            
            
            views.append(dict(
                img=image,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset='MegaDepth',
                label=self.ROOT,
                instance=view_idx))
        return views