# DataLoader
from typing import Any, Dict, Optional, Tuple, List

from scipy.ndimage import affine_transform
import scipy.ndimage
import torch
from lightning import LightningDataModule
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset
import glob

import os
import numpy as np
import cv2

from tqdm import tqdm

scaler = torch.cuda.amp.GradScaler()


class EMdataset3d(torch.utils.data.Dataset):
    def __init__(self,
                 mode: str = "train",
                 transforms: Any = None,
                 crop_shape: Tuple = (128, 128, 128),
                 ):
        self.mode = mode
        self.transforms = transforms
        self.crop_shape = crop_shape
        if self.mode == 'train':
            self.path = "../data/test/masks/**.jpg"
        else:
            self.path = "../data/train/masks/**.jpg"

        self.data = self.load_volume(self.path)
        self.depth = 165
        self.width = 768
        self.height = 1024
        self.dataset_len = int((self.depth * self.width * self.height) / (self.crop_shape[0] * self.crop_shape[1] * self.crop_shape[2]))
        
    # def init_data(self):
    #     self.data = {
    #         x : self.load_volume(self.path)
    #         for x in {"volume", "target"}
    #     }

    def load_volume(self, path: str) -> Dict:
        dataset = sorted(glob.glob(path))

        volume = None
        target = None
        #         print(dataset)
        for z, path in enumerate(tqdm(dataset)):
            mask = (cv2.imread(path, 0) > 127.0) * 1.0
            # path = path.replace(
            #     "labels",
            #     "images",
            # ).replace(".png", ".jp2")
            # if "/kidney_3_dense/" in path:
            #     path = path.replace("kidney_3_dense", "kidney_3_sparse")
            path = path.replace("masks", "images")
            #             print(path)
            image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
            image = np.array(image, dtype=np.uint16)
            if volume is None:
                volume = np.zeros((len(dataset), *image.shape[-2:]), dtype=np.uint16)
                target = np.zeros((len(dataset), *mask.shape[-2:]), dtype=np.uint8)
            volume[z] = image
            target[z] = mask

        return {"volume": volume, "target": target}

    def random_augmentation(self, volume, mask):

        # Random rotation (90-degree increments)
        rotation_axes = [(0, 1), (0, 2), (1, 2)]
        axis = np.random.choice([0, 1, 2])
        angle = np.random.choice([0, 90, 180, 270])
        volume = np.rot90(volume, angle // 90, axes=rotation_axes[axis])
        mask = np.rot90(mask, angle // 90, axes=rotation_axes[axis])

        # Random flips
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=0)
            mask = np.flip(mask, axis=0)
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=1)
            mask = np.flip(mask, axis=1)
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=2)
            mask = np.flip(mask, axis=2)

            # if np.random.rand() > 0.5:
            #     original_shape = volume.shape
            #
            #     zoom_factor = 1 + np.random.uniform(-0.2, 0.2)
            #     volume = zoom(
            #         volume, zoom_factor, order=1
            #     )  # Using bilinear interpolation for volume
            #     mask = zoom(
            #         mask, zoom_factor, order=0
            #     )  # Using nearest-neighbor interpolation for mask
            #
            #     # If zooming out, crop to original size. If zooming in, pad with zeros to original size.
            #     volume, mask = self.match_shape(volume, original_shape), self.match_shape(
            #         mask, original_shape
            #     )

            # if np.random.rand() > 0.3:
            #     brightness_factor = np.random.uniform(
            #         0.8, 1.1
            #     )  # Adjust this range as needed
            #     volume = (volume * brightness_factor).astype(np.uint16)
            #     volume = np.clip(volume, 0, 65535)

        return volume, mask

    def normilize(self, image: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
        image = (image - xmin) / (xmax - xmin)
        image = np.clip(image, 0, 1)
        return image.astype(np.float32)

    def __len__(self):
        return (len(self.data["volume"]))

    def __getitem__(self, item):
        data = self.data
        volume = data["volume"]
        target = data["target"]
        if self.mode == "train":

            sample_new_mask = True
            # Random Crop
            while sample_new_mask:
                start_x = np.random.randint(0, volume.shape[0] - self.crop_shape[0])
                start_y = np.random.randint(0, volume.shape[1] - self.crop_shape[1])
                start_z = np.random.randint(0, volume.shape[2] - self.crop_shape[2])

                volume_crop = volume[
                              start_x: start_x + self.crop_shape[0],
                              start_y: start_y + self.crop_shape[1],
                              start_z: start_z + self.crop_shape[2],
                              ].copy()

                target_crop = target[
                              start_x: start_x + self.crop_shape[0],
                              start_y: start_y + self.crop_shape[1],
                              start_z: start_z + self.crop_shape[2],
                              ].copy()

                # sample_new_mask = sample_non_empty_mask and target_crop.sum() == 0

                volume_crop, target_crop = self.random_augmentation(
                    volume_crop.copy(), target_crop.copy()
                )
        else:
            start_x = np.random.randint(0, volume.shape[0] - self.crop_shape[0])
            start_y = np.random.randint(0, volume.shape[1] - self.crop_shape[1])
            start_z = np.random.randint(0, volume.shape[2] - self.crop_shape[2])

            volume_crop = volume[
                          start_x: start_x + self.crop_shape[0],
                          start_y: start_y + self.crop_shape[1],
                          start_z: start_z + self.crop_shape[2],
                          ].copy()

            target_crop = target[
                          start_x: start_x + self.crop_shape[0],
                          start_y: start_y + self.crop_shape[1],
                          start_z: start_z + self.crop_shape[2],
                          ].copy()

        volume_resized = resize(volume, self.crop_shape, anti_aliasing=True)
        # xmin = np.min(volume_resized)
        # xmax = np.max(volume_resized)
        # print(volume_crop)
        volume_crop = self.normilize(volume_crop, xmin=22, xmax=244)
        #         target_crop = target_crop.astype(np.float32)
        volume_crop, target_crop = np.ascontiguousarray(
            volume_crop
        ), np.ascontiguousarray(target_crop).astype(np.float32)

        return {
            "volume": np.expand_dims(volume_crop, axis=0),
            "target": np.expand_dims(target_crop, axis=0),
            # "id": random_id,
        }

# train_dataset = EMdataset3d(mode = 'train')
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
# print(len(train_loader))
# bar = tqdm(enumerate(train_loader), total = len(train_loader))
#     # for
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# for step, data in bar:
#     img = data["volume"].to(device)
#     target = data["target"].to(device)
#     print(img.shape)


