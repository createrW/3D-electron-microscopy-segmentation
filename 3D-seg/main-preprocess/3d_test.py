import random

import numpy as np
import tifffile as tiff
from glob import glob

from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transform
import torchvision
import cv2
import albumentations as A
import torch
from skimage import io
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root_dir = '../data/'
train_path = root_dir + 'training.tif'
train_gt_path = root_dir + 'training_groundtruth.tif'
test_path = root_dir + 'testing.tif'
test_gt_path = root_dir + 'testing_groundtruth.tif'

class EPFLDataset(Dataset):
    def __init__(self, image_size, train_path, train_gt_path, train=True, tfms=None):
        self.image_size = image_size
        self.train_path = train_path
        self.train_gt_path = train_gt_path
        self.train = train
        self.tfms = tfms

        self.train_stack = io.imread(train_path)
        self.train_gt_stack = io.imread(train_gt_path)

        self.num_stacks, self.x_size, self.y_size = self.train_stack.shape

        self.dataset_len = int(self.num_stacks * ((self.x_size * self.y_size) / (self.image_size ** 2)))
        self.train_len = int(0.8 * self.dataset_len)
        self.val_len = self.dataset_len - self.train_len

        idx_list = list(range(self.dataset_len))
        random.Random(42).shuffle(idx_list)

        self.train_idxs = idx_list[:self.train_len]
        self.val_idxs = idx_list[self.train_len:]

    def __len__(self):
        if self.train:
            return self.train_len
        else:
            return self.val_len

    def __getitem__(self, idx):
        image_num = self.train_idxs[idx] if self.train else self.val_idxs[idx]
        images_single_stack = (self.x_size * self.y_size) / (self.image_size * self.image_size)
        num_rows = self.x_size / self.image_size
        num_cols = self.y_size / self.image_size
        stack_num = int(image_num / images_single_stack)
        image_num_in_stack = image_num % (images_single_stack)
        row_num = int(image_num_in_stack / num_cols)
        col_num = image_num_in_stack % num_rows

        start_row = int(row_num * self.image_size)
        end_row = int(start_row + self.image_size)

        start_col = int(col_num * self.image_size)
        end_col = int(start_col + self.image_size)

        img = self.train_stack[stack_num, start_row:end_row, start_col:end_col]
        img = img.reshape(-1, 256, 256)
        # img = np.expand_dims(img, -1)
        img = img.astype('float32')
        img = torch.from_numpy(img)

        mask = self.train_gt_stack[stack_num, start_row:end_row, start_col:end_col]
        mask = mask.reshape(-1, 256, 256)
        # mask = np.expand_dims(mask, -1)
        mask = np.where(mask == 255, 1, 0)
        mask = mask.astype('float32')
        mask = torch.from_numpy(mask)
        mask = mask.type(torch.LongTensor)

        ret_value = {
            'image': img,
            'mask': mask
        }

        return ret_value

transform_images = A.Compose([
        # A.resize(),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, p=0.7),
        A.OneOf([
                A.GaussNoise(var_limit = (10.0, 50.0)),
                A.GaussianBlur(),
                A.MotionBlur(),
                A.MedianBlur(blur_limit=3),
                ], p=0.4),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.OpticalDistortion(distort_limit=1., p=1.0)
        ],p=0.2),
        A.ShiftScaleRotate(p=0.7, scale_limit=0.5, shift_limit=0.2, rotate_limit=30),
        A.CoarseDropout(max_holes=1, max_height=0.25, max_width=0.25),
        ToTensorV2(transpose_mask=True)
    ])

transform_masked = A.Compose([
    ToTensorV2(transpose_mask=True)
])

