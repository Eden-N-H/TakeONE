import torch
from torch.utils.data import Dataset
import cv2 as cv
import numpy as np
import glob
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from .image_util import crop, random_flip, random_rotate

from torchvision import transforms
from PIL import ImageFilter
import random

class RandomMotionBlur:
    def __init__(self, p=0.5, radius_range=(1.0, 3.0)):
        self.p = p
        self.radius_range = radius_range
    def __call__(self, img):
        if random.random() < self.p:
            r = random.uniform(*self.radius_range)
            return img.filter(ImageFilter.GaussianBlur(radius=r))
        return img

class SRDataset(Dataset):
    """
    Custom PyTorch Dataset for Super-Resolution.
    This class is responsible for loading low-resolution (LR) and
    high-resolution (HR) image pairs, and applying data augmentation.
    """
    def __init__(self, lr_root, hr_root, scale, hr_crop_size=192):
        """
        :param lr_root: The root directory containing low-resolution images.
        :param hr_root: The root directory containing high-resolution images.
        :param scale: The scale factor (e.g., 2, 3, or 4).
        :param hr_crop_size: The desired crop size for high-resolution images.
        """
        super(SRDataset, self).__init__()
        self.lr_files = sorted(glob.glob(os.path.join(lr_root, '*')))
        self.hr_files = sorted(glob.glob(os.path.join(hr_root, '*')))
        self.scale = scale
        self.hr_crop_size = hr_crop_size

        # Ensure that the number of LR and HR files match
        assert len(self.lr_files) == len(self.hr_files), "Number of LR and HR images do not match."

    def __len__(self):
        """
        Returns the total number of image pairs in the dataset.
        """
        return len(self.lr_files)

    def __getitem__(self, idx):
        """
        Loads and preprocesses a single image pair.
        
        :param idx: The index of the image pair to retrieve.
        :return: A tuple containing the low-resolution and high-resolution tensors.
        """
        lr_img = cv.imread(self.lr_files[idx], cv.IMREAD_COLOR)
        hr_img = cv.imread(self.hr_files[idx], cv.IMREAD_COLOR)

        # Ensure images were loaded
        if lr_img is None or hr_img is None:
            raise FileNotFoundError(f"Failed to read {self.lr_files[idx]} or {self.hr_files[idx]}")

        # Convert BGR -> RGB
        lr_img = cv.cvtColor(lr_img, cv.COLOR_BGR2RGB)
        hr_img = cv.cvtColor(hr_img, cv.COLOR_BGR2RGB)

        # Convert to float32 and normalize to [0, 1]
        lr_img = lr_img.astype(np.float32) / 255.0
        hr_img = hr_img.astype(np.float32) / 255.0


        # Apply cropping and augmentation using the functions from image_util
        lr_img_cropped, hr_img_cropped = crop(lr_img, hr_img, data_type='array',
                                               hr_crop_size=self.hr_crop_size, scale=self.scale)
        lr_img_flipped, hr_img_flipped = random_flip(lr_img_cropped, hr_img_cropped,
                                                     data_type='array')
        lr_img_aug, hr_img_aug = random_rotate(lr_img_flipped, hr_img_flipped,
                                               data_type='array')

        # Convert from HxWxC to CxHxW format and then to a PyTorch tensor
        lr_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(lr_img_aug, (2, 0, 1)))).float()
        hr_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(hr_img_aug, (2, 0, 1)))).float()


        return lr_tensor, hr_tensor
