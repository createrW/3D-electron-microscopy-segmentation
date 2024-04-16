import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from skimage import io

import cv2
from PIL import Image
import random

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.v2 import RandomResize
from torchvision.utils import make_grid, save_image
import torchvision.transforms.functional as TF
from torchvision import transforms
import albumentations as A

root_dir = '../data/'
train_path = root_dir + 'training.tif'
train_gt_path = root_dir + 'training_groundtruth.tif'
test_path = root_dir + 'testing.tif'
test_gt_path = root_dir + 'testing_groundtruth.tif'

#Dataset & DataLoader

#Train &  VAL
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

    def __shape__(self):
        return self.train_stack.shape


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



##Test
class EPFLTestDataset(Dataset):
    def __init__(self, image_size, train_path, train_gt_path, tfms=None):
        self.image_size = image_size
        self.train_path = train_path
        self.train_gt_path = train_gt_path
        self.tfms = tfms

        self.train_stack = io.imread(train_path)
        self.train_gt_stack = io.imread(train_gt_path)

        self.num_stacks, self.x_size, self.y_size = self.train_stack.shape

        self.dataset_len = int(self.num_stacks * ((self.x_size * self.y_size) / (self.image_size ** 2)))

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        image_num = idx
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
##
tfms_img = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.Affine(scale={"x": (0.7, 1.3), "y": (0.7, 1.3)}, translate_percent={"x": (0, 0.1), "y": (0, 0.1)},
                     rotate=(-30, 30), shear=(-20, 20), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.5),
            A.OneOf([
                A.Blur(blur_limit=3, p=0.2),
                A.MedianBlur(blur_limit=3, p=0.2),
            ], p=1.0),
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=10, border_mode=1, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.1, border_mode=1, p=0.5)
            ], p=0.4),
            A.OneOf([
                A.Resize(256, 256, cv2.INTER_LINEAR, p=1),
                A.Compose([
                    RandomResize(256, 256),
                    A.PadIfNeeded(256, 256, position="random", border_mode=cv2.BORDER_REPLICATE, p=1.0),
                    A.RandomCrop(256, 256, p=1.0)
                ], p=1.0),
            ], p=1.0),
            A.GaussNoise(var_limit=0.05, p=0.2),
        ])
tfms_mask = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.Affine(scale={"x": (0.7, 1.3), "y": (0.7, 1.3)}, translate_percent={"x": (0, 0.1), "y": (0, 0.1)},
                     rotate=(-30, 30), shear=(-20, 20), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.5),
            A.OneOf([
                A.Blur(blur_limit=3, p=0.2),
                A.MedianBlur(blur_limit=3, p=0.2),
            ], p=1.0),
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=10, border_mode=1, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.1, border_mode=1, p=0.5)
            ], p=0.4),
            A.OneOf([
                A.Resize(256, 256, cv2.INTER_LINEAR, p=1),
                A.Compose([
                    RandomResize(256, 256),
                    A.PadIfNeeded(256, 256, position="random", border_mode=cv2.BORDER_REPLICATE, p=1.0),
                    A.RandomCrop(256, 256, p=1.0)
                ], p=1.0),
            ], p=1.0),
            A.GaussNoise(var_limit=0.05, p=0.2),
        ])

tfms = {
    'img':tfms_img,
    'mask':tfms_mask
}

image_size = 256
batch = 4

train_dataset = EPFLDataset(image_size, train_path, train_gt_path, train=True, tfms=tfms)
val_dataset = EPFLDataset(image_size, train_path, train_gt_path, train=False, tfms=tfms)
test_dataset = EPFLTestDataset(image_size, test_path, test_gt_path, tfms=tfms)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True, num_workers=2)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_dataset, batch_size=int(batch/2))

print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))

print(train_dataset.__shape__())


def collate_fn(batch):
    images = [d['image'] for d in batch]
    max_depth = max([image.shape[0] for image in images])

    batch_images, batch_masks = [], []
    for i in range(len(batch)):
        cur_depth = images[i].shape[0]
        width, height = images[i].shape[1:]
        padding_depth = max_depth - cur_depth

        padding_image = torch.zeros((padding_depth, width, height))
        image = torch.cat([torch.tensor(images[i]), padding_image], dim=0)
        batch_images.append(image)

        if 'mask' in batch[0].keys():
            padding_mask = torch.zeros((4, padding_depth, width, height))
            mask = torch.cat([torch.tensor(batch[i]['mask']), padding_mask], dim=0)
            batch_masks.append(mask)

    item = {
        'image': torch.stack(batch_images, dim=0)
    }

    if 'mask' in batch[0].keys():
        item['mask'] = torch.stack(batch_masks, dim=0)

    if 'label' in batch[0].keys():
        batch_label = torch.LongTensor([d['label'] for d in batch])
        item['label'] = batch_label
    print(item['image'].shape, item['mask'].shape)
    return item