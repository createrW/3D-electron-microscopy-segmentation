import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import gc
import cv2
import math
import copy
import time
import random
import glob
from matplotlib import pyplot as plt

# For data manipulation
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
# from Vnet3D.net3 import *
# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
import torchvision
import torchmetrics
# Utils
import joblib
from torchmetrics.classification import BinaryJaccardIndex
from tqdm import tqdm
from collections import defaultdict
# from preprocess.dataset3d import EMdataset3d
# Sklearn Imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold

# For Image Models
import timm

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"the device is on {device}")
import math
import os
import numpy as np
from skimage.io import imread
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from biapy.utils.util import load_3d_images_from_dir, order_dimensions


def load_and_prepare_3D_data(train_path, train_mask_path, cross_val=False, cross_val_nsplits=5, cross_val_fold=1,
                             val_split=0.1, seed=0, shuffle_val=True, crop_shape=(80, 80, 80, 1), y_upscaling=(1, 1, 1),
                             random_crops_in_DA=False,
                             ov=(0, 0, 0), padding=(0, 0, 0), minimum_foreground_perc=-1,
                             reflect_to_complete_shape=False, convert_to_rgb=False,
                             preprocess_cfg=None, is_y_mask=False, preprocess_f=None):
    """
    Load train and validation images from the given paths to create 3D data.

    Parameters
    ----------
    train_path : str
        Path to the training data.

    train_mask_path : str
        Path to the training data masks.

    cross_val : bool, optional
        Whether to use cross validation or not.

    cross_val_nsplits : int, optional
        Number of folds for the cross validation.

    cross_val_fold : int, optional
        Number of the fold to be used as validation.

    val_split : float, optional
        ``%`` of the train data used as validation (value between ``0`` and ``1``).

    seed : int, optional
        Seed value.

    shuffle_val : bool, optional
        Take random training examples to create validation data.

    crop_shape : 4D tuple
        Shape of the train subvolumes to create. E.g. ``(z, y, x, channels)``.

    y_upscaling : Tuple of 3 ints, optional
        Upscaling to be done when loading Y data. Use for super-resolution workflow.

    random_crops_in_DA : bool, optional
        To advice the method that not preparation of the data must be done, as random subvolumes will be created on
        DA, and the whole volume will be used for that.

    ov : Tuple of 3 floats, optional
        Amount of minimum overlap on x, y and z dimensions. The values must be on range ``[0, 1)``, that is, ``0%``
        or ``99%`` of overlap. E. g. ``(z, y, x)``.

    padding : Tuple of ints, optional
        Size of padding to be added on each axis ``(z, y, x)``. E.g. ``(24, 24, 24)``.

    minimum_foreground_perc : float, optional
        Minimum percetnage of foreground that a sample need to have no not be discarded.

    reflect_to_complete_shape : bool, optional
        Wheter to increase the shape of the dimension that have less size than selected patch size padding it with
        'reflect'.

    self_supervised_args : dict, optional
        Arguments to create ground truth data for self-supervised workflow.

    convert_to_rgb : bool, optional
        In case RGB images are expected, e.g. if ``crop_shape`` channel is 3, those images that are grayscale are
        converted into RGB.

    preprocess_cfg : dict, optional
        Configuration parameters for preprocessing, is necessary in case you want to apply any preprocessing.

    is_y_mask : bool, optional
        Whether the data are masks. It is used to control the preprocessing of the data.

    preprocess_f : function, optional
        The preprocessing function, is necessary in case you want to apply any preprocessing.

    Returns
    -------
    X_train : 5D Numpy array
        Train images. E.g. ``(num_of_images, z, y, x, channels)``.

    Y_train : 5D Numpy array
        Train images' mask. E.g. ``(num_of_images, z, y, x, channels)``.

    X_val : 5D Numpy array, optional
        Validation images (``val_split > 0``). E.g. ``(num_of_images, z, y, x, channels)``.

    Y_val : 5D Numpy array, optional
        Validation images' mask (``val_split > 0``). E.g. ``(num_of_images, z, y, x, channels)``.

    filenames : List of str
        Loaded train filenames.

    Examples
    --------
    ::

        # EXAMPLE 1
        # Case where we need to load the data and creating a validation split
        train_path = "data/train/x"
        train_mask_path = "data/train/y"

        # Train data is (15, 91, 1024, 1024) where (number_of_images, z, y, x), so each image shape should be this:
        img_train_shape = (91, 1024, 1024, 1)
        # 3D subvolume shape needed
        train_3d_shape = (40, 256, 256, 1)

        X_train, Y_train, X_val,
        Y_val, filenames = load_and_prepare_3D_data_v2(train_path, train_mask_path, train_3d_shape,
                                                        val_split=0.1, shuffle_val=True, ov=(0,0,0))

        # The function will print the shapes of the generated arrays. In this example:
        #     *** Loaded train data shape is: (315, 40, 256, 256, 1)
        #     *** Loaded train mask shape is: (315, 40, 256, 256, 1)
        #     *** Loaded validation data shape is: (35, 40, 256, 256, 1)
        #     *** Loaded validation mask shape is: (35, 40, 256, 256, 1)
        #
    """

    print("### LOAD ###")

    # Disable crops when random_crops_in_DA is selected
    if random_crops_in_DA:
        crop = False
    else:
        if cross_val:
            crop = False
            # Delay the crop to be made after cross validation
            delay_crop = True
        else:
            crop = True
            delay_crop = False

            # Check validation
    if val_split > 0 or cross_val:
        create_val = True
    else:
        create_val = False

    print("0) Loading train images . . .")
    X_train, _, _, t_filenames = load_3d_images_from_dir(train_path, crop=crop, crop_shape=crop_shape,
                                                         overlap=ov, padding=padding, return_filenames=True,
                                                         reflect_to_complete_shape=reflect_to_complete_shape,
                                                         convert_to_rgb=convert_to_rgb, preprocess_cfg=preprocess_cfg,
                                                         is_mask=False, preprocess_f=preprocess_f)

    if train_mask_path is not None:
        print("1) Loading train GT . . .")
        scrop = (
        crop_shape[0] * y_upscaling[0], crop_shape[1] * y_upscaling[1], crop_shape[2] * y_upscaling[2], crop_shape[3])
        Y_train, _, _ = load_3d_images_from_dir(train_mask_path, crop=crop, crop_shape=scrop, overlap=ov,
                                                padding=padding, reflect_to_complete_shape=reflect_to_complete_shape,
                                                check_channel=False, check_drange=False,
                                                preprocess_cfg=preprocess_cfg, is_mask=is_y_mask,
                                                preprocess_f=preprocess_f)
    else:
        Y_train = None

    #     print(X_train.shape)
    #     print(X_train)
    if isinstance(X_train, list):
        raise NotImplementedError("If you arrived here means that your images are not all of the same shape, and you "
                                  "select DATA.EXTRACT_RANDOM_PATCH = True, so no crops are made to ensure all images "
                                  "have the same shape. Please, crop them into your DATA.PATCH_SIZE and run again (you "
                                  "can use one of the script from here to crop: https://github.com/BiaPyX/BiaPy/tree/master/biapy/utils/scripts)")

    # Discard images that do not surpass the foreground percentage threshold imposed
    if minimum_foreground_perc != -1 and Y_train is not None:
        print("Data that do not have {}% of foreground is discarded".format(minimum_foreground_perc))

        X_train_keep = []
        Y_train_keep = []
        are_lists = True if type(Y_train) is list else False

        samples_discarded = 0
        for i in tqdm(range(len(Y_train)), leave=False):
            labels, npixels = np.unique((Y_train[i] > 0).astype(np.uint8), return_counts=True)

            total_pixels = 1
            for val in list(Y_train[i].shape):
                total_pixels *= val

            discard = False
            if len(labels) == 1:
                discard = True
            else:
                if (sum(npixels[1:] / total_pixels)) < minimum_foreground_perc:
                    discard = True
            print(discard)
            if discard:
                samples_discarded += 1
            else:
                if are_lists:
                    X_train_keep.append(X_train[i])
                    Y_train_keep.append(Y_train[i])
                else:
                    X_train_keep.append(np.expand_dims(X_train[i], 0))
                    Y_train_keep.append(np.expand_dims(Y_train[i], 0))
        del X_train, Y_train

        if len(X_train_keep) == 0:
            raise ValueError(
                "'TRAIN.MINIMUM_FOREGROUND_PER' value is too high, leading to the discarding of all training samples. Please, "
                "reduce its value.")

        if not are_lists:
            X_train_keep = np.concatenate(X_train_keep)
            Y_train_keep = np.concatenate(Y_train_keep)

        # Rename
        X_train, Y_train = X_train_keep, Y_train_keep
        del X_train_keep, Y_train_keep

        print("{} samples discarded!".format(samples_discarded))
        if type(Y_train) is not list:
            print("*** Remaining data shape is {}".format(X_train.shape))
            if X_train.shape[0] <= 1 and create_val:
                raise ValueError("0 or 1 sample left to train, which is insufficent. "
                                 "Please, decrease the percentage to be more permissive")
        else:
            print("*** Remaining data shape is {}".format((len(X_train),) + X_train[0].shape[1:]))
            if len(X_train) <= 1 and create_val:
                raise ValueError("0 or 1 sample left to train, which is insufficent. "
                                 "Please, decrease the percentage to be more permissive")

    if Y_train is not None and len(X_train) != len(Y_train):
        raise ValueError("Different number of raw and ground truth items ({} vs {}). "
                         "Please check the data!".format(len(X_train), len(Y_train)))

    # Create validation data splitting the train
    if create_val:
        print("Creating validation data")
        Y_val = None
        if not cross_val:
            if Y_train is not None:
                X_train, X_val, Y_train, Y_val = train_test_split(
                    X_train, Y_train, test_size=val_split, shuffle=shuffle_val, random_state=seed)
            else:
                X_train, X_val = train_test_split(
                    X_train, test_size=val_split, shuffle=shuffle_val, random_state=seed)
        else:
            skf = StratifiedKFold(n_splits=cross_val_nsplits, shuffle=shuffle_val,
                                  random_state=seed)
            fold = 1
            train_index, test_index = None, None

            y_len = len(Y_train) if Y_train is not None else len(X_train)
            for t_index, te_index in skf.split(np.zeros(len(X_train)), np.zeros(y_len)):
                if cross_val_fold == fold:
                    X_train, X_val = X_train[t_index], X_train[te_index]
                    if Y_train is not None:
                        Y_train, Y_val = Y_train[t_index], Y_train[te_index]
                    train_index, test_index = t_index.copy(), te_index.copy()
                    break
                fold += 1

            if len(test_index) > 5:
                print("Fold number {}. Printing the first 5 ids: {}".format(fold, test_index[:5]))
            else:
                print("Fold number {}. Indexes used in cross validation: {}".format(fold, test_index))

            # Then crop after cross validation
            if delay_crop:
                # X_train
                data = []
                for img_num in range(len(X_train)):
                    if X_train[img_num].shape != crop_shape[:3] + (X_train[img_num].shape[-1],):
                        img = X_train[img_num]
                        img = crop_3D_data_with_overlap(
                            X_train[img_num][0] if isinstance(X_train, list) else X_train[img_num],
                            crop_shape[:3] + (X_train[img_num].shape[-1],), overlap=ov, padding=padding, verbose=False)
                    data.append(img)
                X_train = np.concatenate(data)
                del data

                # Y_train
                if Y_train is not None:
                    data_mask = []
                    scrop = (crop_shape[0], crop_shape[1] * y_upscaling[0], crop_shape[2] * y_upscaling[1],
                             crop_shape[3] * y_upscaling[2])
                    for img_num in range(len(Y_train)):
                        if Y_train[img_num].shape != scrop[:3] + (Y_train[img_num].shape[-1],):
                            img = Y_train[img_num]
                            img = crop_3D_data_with_overlap(
                                Y_train[img_num][0] if isinstance(Y_train, list) else Y_train[img_num],
                                scrop[:3] + (Y_train[img_num].shape[-1],), overlap=ov, padding=padding, verbose=False)
                        data_mask.append(img)
                    Y_train = np.concatenate(data_mask)
                    del data_mask

                # X_val
                data = []
                for img_num in range(len(X_val)):
                    if X_val[img_num].shape != crop_shape[:3] + (X_val[img_num].shape[-1],):
                        img = X_val[img_num]
                        img = crop_3D_data_with_overlap(
                            X_val[img_num][0] if isinstance(X_val, list) else X_val[img_num],
                            crop_shape[:3] + (X_val[img_num].shape[-1],), overlap=ov, padding=padding, verbose=False)
                    data.append(img)
                X_val = np.concatenate(data)
                del data

                # Y_val
                if Y_val is not None:
                    data_mask = []
                    scrop = (crop_shape[0], crop_shape[1] * y_upscaling[0], crop_shape[2] * y_upscaling[1],
                             crop_shape[3] * y_upscaling[2])
                    for img_num in range(len(Y_val)):
                        if Y_val[img_num].shape != scrop[:3] + (Y_val[img_num].shape[-1],):
                            img = Y_val[img_num]
                            img = crop_3D_data_with_overlap(
                                Y_val[img_num][0] if isinstance(Y_val, list) else Y_val[img_num],
                                scrop[:3] + (Y_val[img_num].shape[-1],), overlap=ov, padding=padding, verbose=False)
                        data_mask.append(img)
                    Y_val = np.concatenate(data_mask)
                    del data_mask

    # Convert the original volumes as they were a unique subvolume
    if random_crops_in_DA and X_train.ndim == 4:
        print(X_train.shape)
        # print(X)
        X_train = np.expand_dims(X_train, axis=0)
        if Y_train is not None:
            Y_train = np.expand_dims(Y_train, axis=0)
        if create_val:
            X_val = np.expand_dims(X_val, axis=0)
            if Y_val is not None:
                Y_val = np.expand_dims(Y_val, axis=0)

    if create_val:
        print("*** Loaded train data shape is: {}".format(X_train.shape))
        if Y_train is not None:
            print("*** Loaded train GT shape is: {}".format(Y_train.shape))
        print("*** Loaded validation data shape is: {}".format(X_val.shape))
        if Y_val is not None:
            print("*** Loaded validation GT shape is: {}".format(Y_val.shape))
        if not cross_val:
            return X_train, Y_train, X_val, Y_val, t_filenames
        else:
            return X_train, Y_train, X_val, Y_val, t_filenames, test_index
    else:
        print("*** Loaded train data shape is: {}".format(X_train.shape))
        if Y_train is not None:
            print("*** Loaded train GT shape is: {}".format(Y_train.shape))
        return X_train, Y_train, t_filenames





train_img, train_mask, val_img, val_mask, c = load_and_prepare_3D_data("../data/tr/img", "../data/tr/mask", crop_shape=(80, 80, 80, 1))
test_img, test_mask, _ =  load_and_prepare_3D_data("../data/te/img", "../data/te/mask", crop_shape=(80, 80, 80, 1), cross_val=False, cross_val_nsplits=0, cross_val_fold=0,
    val_split=0)

train_img = train_img.reshape(351, 1, 80, 80, 80)
train_mask = train_mask.reshape(351, 1, 80, 80, 80)
val_img = val_img.reshape(39, 1, 80, 80, 80)
val_mask = val_mask.reshape(39, 1, 80, 80, 80)
test_img = test_img.reshape(390, 1, 80, 80, 80)
test_mask = test_mask.reshape(390, 1, 80, 80, 80)
print(f"train_img shape is {train_img.shape}")
print(f"val_img shape is {val_img.shape}")
print(f"test_img shape is {test_img.shape}")


class EmDataset(Dataset):
    def __init__(self, image, target, transforms=None):
        self.transforms = transforms
        self.target = target
        self.image = image

    def __len__(self):
        return (len(self.image))

    def __getitem__(self, idx):
        image = self.image[idx]
        target = self.target[idx]
        mean = np.mean(image)
        std = np.std(image)
        image = (image - mean) / std
        image = image.astype(np.float64)
        #         target = np.where((target > 127), 1.0, 0.0).astype(np.float64)
        target = np.where((target > 127), 1.0, 0.0)
        #         print(f"image:{image}")
        #         print(f"target:{target}")
        #         target = np.where()
        if self.transforms:
            image = self.transforms(image=image, mask=target)["image"]
            image = image.copy()
            mask = self.transforms(image=image, mask=target)["mask"]
        # return (image.to(device), target.to(device))
        return {
            'image': torch.from_numpy(image).to(device),
            'mask': torch.from_numpy(target).to(device)
        }




train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
#             A.RandomBrightnessContrast(brightness_limit=0.1, p=0.7),
#             A.CoarseDropout(max_holes=1, max_height=0.25, max_width=0.25)
        ])
val_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)
        ])
train_dataset = EmDataset(train_img, train_mask, transforms = train_transform)
train_dataloader = DataLoader(train_dataset, batch_size = 1, shuffle = True, drop_last = True)
val_dataset = EmDataset(val_img, train_mask, transforms = val_transform)
val_dataloader = DataLoader(val_dataset, batch_size = 1, shuffle = True, drop_last = True)
test_dataset = EmDataset(test_img, test_mask, transforms = None)
test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = True, drop_last = True)



#show some photos
import matplotlib.pyplot as plt
import numpy as np
dataset = train_dataset
# ax = plt.axes(projection='3d')
# # ax.plot3D(x, y, z,'gray')
# for i in range(len(dataset)):
i = np.random.randint(0,len(dataset))
# for i in range(0,len(dataset)):
data = dataset[i]
image = data["image"]
mask = data["mask"]
print(image.shape)
image = torch.squeeze(image, 0)
mask = torch.squeeze(mask, 0)
image = image.cpu().numpy()
mask = mask.cpu().numpy()
# print(mask)
# #     print(image)
# #     print(mask)
# print(image.shape)
# for j in range(80):
# plt.subplot(121)
# ax.subplot(121)
# ax.plot3D(image)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# print(image[64, :, :].shape)
axes[0].imshow(image[64, :, :], cmap='bone')
axes[0].set_title('YZ Plane')

axes[1].imshow(image[:, 64, :], cmap='bone')
axes[1].set_title('XZ Plane')

axes[2].imshow(image[:, :, 64], cmap='bone')
axes[2].set_title('XY Plane')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(mask[64, :, :], cmap='gray')
axes[0].set_title('YZ Plane')

axes[1].imshow(mask[:, 64, :], cmap='gray')
axes[1].set_title('XZ Plane')

axes[2].imshow(mask[:, :, 64], cmap='gray')
axes[2].set_title('XY Plane')



#build the net
import rasterio
from rasterio.windows import Window
import monai
model = UNet3D(in_channels = 1, num_classes = 1)
# from monai.networks.nets import UNet
# model = UNet(
#     spatial_dims=3,
#     in_channels=1,
#     out_channels=1,
#     channels=(16, 32, 64, 128, 256),
#     strides=(2, 2, 2, 2),
#     num_res_units=2,
# )
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(),
                  lr=1e-4, weight_decay=1e-3)

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-3, -2, -1)):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()
        return 1 - dc


bce_fn = nn.BCEWithLogitsLoss()
dice_fn = SoftDiceLoss()


def loss_fn(y_pred, y_true):
    bce = bce_fn(y_pred, y_true)
    dice = dice_fn(y_pred.sigmoid(), y_true)
    return dice





