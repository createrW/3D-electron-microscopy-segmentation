import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
from skimage import io

import random
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image
import torchvision.transforms.functional as TF
from torchvision import transforms
import albumentations as A
import tifffile
train_path ='../data/tr/img/training.tif'
train_gt_path = '../data/tr/mask/training_groundtruth.tif'
test_path = '../data/te/img/testing.tif'
test_gt_path = '../data/te/mask/testing_groundtruth.tif'
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
crop_shape = (128, 128, 128)
batch_size = 1
epoch = 10
# optimizer = torch.optim.Adam()
def normilize(image: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    image = (image - xmin) / (xmax - xmin)
    image = np.clip(image, 0, 1)
    return image.astype(np.float32)

def random_augmentation(volume):

    # Random rotation (90-degree increments)
    rotation_axes = [(0, 1), (0, 2), (1, 2)]
    axis = np.random.choice([0, 1, 2])
    angle = np.random.choice([0, 90, 180, 270])
    volume = np.rot90(volume, angle // 90, axes=rotation_axes[axis])
    # mask = np.rot90(mask, angle // 90, axes=rotation_axes[axis])

    # Random flips
    if np.random.rand() > 0.5:
        volume = np.flip(volume, axis=0)
        # mask = np.flip(mask, axis=0)
    if np.random.rand() > 0.5:
        volume = np.flip(volume, axis=1)
        # mask = np.flip(mask, axis=1)
    if np.random.rand() > 0.5:
        volume = np.flip(volume, axis=2)
        # mask = np.flip(mask, axis=2)



    return volume
def crop_3d(crop_shape, tif_path, train = False, is_img = False):
    # Calculate the number of 3D slices per dimension
    tif_image = tifffile.imread(tif_path)
    num_slices_x = tif_image.shape[2] // crop_shape[0]
    num_slices_y = tif_image.shape[1] // crop_shape[1]
    num_slices_z = tif_image.shape[0] // crop_shape[2]

    # Initialize an empty array to store the 3D slices
    if is_img:
        slices_3d = np.zeros((num_slices_x * num_slices_y * num_slices_z, crop_shape[0], crop_shape[1], crop_shape[2]), dtype=np.float32)
    else:
        slices_3d = np.zeros((num_slices_x * num_slices_y * num_slices_z, crop_shape[0], crop_shape[1], crop_shape[2]),
                             dtype=np.uint16)
    # Fill the array with the 3D slices
    idx = 0
    for z in range(num_slices_z):
        for y in range(num_slices_y):
            for x in range(num_slices_x):
                start_x = x * crop_shape[0]
                end_x = start_x + crop_shape[0]
                start_y = y * crop_shape[1]
                end_y = start_y + crop_shape[1]
                start_z = z * crop_shape[2]
                end_z = start_z + crop_shape[2]
                slice_3d = tif_image[start_z:end_z, start_y:end_y, start_x:end_x]
                if train:
                    # slice_3d = random_augmentation(slice_3d)
                    pass
                # print(f"Slice shape for ({start_x}:{end_x}, {start_y}:{end_y}, {start_z}:{end_z}): {slice_3d.shape}")
                if slice_3d.shape == crop_shape:
                    # print(f"Slice shape for ({start_x}:{end_x}, {start_y}:{end_y}, {start_z}:{end_z}): {slice_3d.shape}")
                    slices_3d[idx] = slice_3d
                    idx += 1
    return slices_3d
train_img = crop_3d(crop_shape, train_path, train=True, is_img=True)
train_mask = crop_3d(crop_shape, train_gt_path, True, False)
test_img = crop_3d(crop_shape, test_path, False, True)
test_mask = crop_3d(crop_shape, test_gt_path, False, False)

class EleMic1(Dataset):
  def __init__(self,image, target, transforms_image = None, transform_mask = None):
    self.transform_image=transforms_image
    self.transform_mask = transform_mask
    self.target=target
    self.image=image

  def __len__(self):
    return(len(self.image))

  def __getitem__(self, idx):
    image = self.image[idx]
    target = self.target[idx]
    target = np.where(target == 255, 1, 0)
    target = target.astype('float32')
    image = image.astype("float32")
    image /= 255
    # if self.transform_image:
    #     image = self.transforms(image=image, mask=target)["image"]
    #     # image = image.copy()
    #     mask = self.transforms(image=image, mask=target)["mask"]
    if self.transform_image != None:
        target = self.transform_image(image)
    if self.transform_mask != None:
        target = self.transform_mask(target)
    # return (image.to(device), target.to(device))
    return {
        'image': torch.from_numpy(image).to(device).squeeze(),
        'mask': torch.from_numpy(target).to(device).squeeze()
    }

train_dataset = EleMic1(train_img, train_mask, None, None)
test_dataset = EleMic1(test_img, test_mask, None, None)
train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

# #test
# for step, data in tqdm(enumerate(train_dataloader), total = len(train_dataloader)):
#     img, mask = data["image"], data["mask"]

#build the net
from my_3D.unet3d_2 import *
model = UNet(1, 1, 1)
# from innovation_models.densenet3d import *
# from innovation_models.densenet3dblock import *
# model = DenseUNet3d()
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(),
                  lr=1e-4, weight_decay=1e-3)
# from preprocess.loss_3d import *
# loss_fn = DiceLoss()
class BoundaryDoULoss3D(nn.Module):
    def __init__(self, n_classes=1):
        super(BoundaryDoULoss3D, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _adaptive_size(self, score, target):
        kernel = (
            torch.Tensor(
                [
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                ]
            )
            .to(target.device)
            .half()
        )

        padding_out = torch.zeros(
            (
                target.shape[0],
                target.shape[-3] + 2,
                target.shape[-2] + 2,
                target.shape[-1] + 2,
            )
        )
        padding_out[:, 1:-1, 1:-1, 1:-1] = target
        d, h, w = 3, 3, 3

        Y = torch.zeros(
            (
                padding_out.shape[0],
                padding_out.shape[1] - d + 1,
                padding_out.shape[2] - h + 1,
                padding_out.shape[3] - w + 1,
            )
        ).to(
            target.device
        )  # .cuda()

        for i in range(Y.shape[0]):
            Y[i, :, :, :] = torch.nn.functional.conv3d(
                target[i].unsqueeze(0).unsqueeze(0).half(),
                kernel.unsqueeze(0).unsqueeze(0),  # .cuda(),
                padding=1,
            )

        Y = Y * target
        Y[Y == 7] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(alpha, 0.8)  # We recommend using a truncated alpha of 0.8.

        loss = (z_sum + y_sum - 2 * intersect + smooth) / (
            z_sum + y_sum - (1 + alpha) * intersect + smooth
        )

        return loss

    def forward(self, inputs, target):
        inputs = inputs.sigmoid()

        assert (
            inputs.size() == target.size()
        ), "predict {} & target {} shape do not match".format(
            inputs.size(), target.size()
        )

        # return self._adaptive_size(inputs, target)
        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._adaptive_size(inputs[:, i], target[:, i])
        return loss / self.n_classes

class UncertaintyEstimationLoss3D(nn.Module):
    def __init__(self, num_samples=10):
        super(UncertaintyEstimationLoss3D, self).__init__()
        self.base_criterion = BoundaryDoULoss3D()
        self.num_samples = num_samples

    def forward(self, outputs, targets):
        # outputs: tensor of shape (batch_size, num_classes, depth, height, width)
        # targets: tensor of shape (batch_size, 1, depth, height, width), binary segmentation mask

        total_loss = 0.0

        for _ in range(self.num_samples):
            # Apply dropout during inference
            sampled_outputs = F.dropout3d(outputs, training=True)

            # Calculate loss using the base criterion (e.g., binary cross entropy)
            loss = self.base_criterion(sampled_outputs, targets)
            total_loss += loss

        # Compute mean loss over all samples
        mean_loss = total_loss / self.num_samples

        # Compute uncertainty as the variance of the sampled losses
        uncertainty = torch.var(
            torch.stack(
                [
                    self.base_criterion(F.dropout3d(outputs, training=True), targets)
                    for _ in range(self.num_samples)
                ]
            ),
            dim=0,
        )

        # Total loss is a combination of mean loss and uncertainty
        total_loss = mean_loss + uncertainty

        return total_loss

# loss_fn = BoundaryDoULoss3D()
# model.load_state_dict(torch.load("./model_best_3d.pth"))
# # training
# @torch.no_grad()
# def validation(model, loader, loss_fn):
#     losses = []
#     model.eval()
#     bar = tqdm(enumerate(loader), total = len(loader))
#     for step, data in bar:
#         image, target = data["image"].to(device), data["mask"].float().to(device)
#         image = torch.unsqueeze(image, 1)
#         target = torch.unsqueeze(target, 1)
#         # image, target = image.to(device), target.float().to(device)
#         output = model(image)
#         loss = loss_fn(output, target)
#         losses.append(loss.item())
#
#     return np.array(losses).mean()
#
# header = r'''
#         Train | Valid
# Epoch |  Loss |  Loss | Time, m
# '''
# #          Epoch         metrics            time
# # raw_line = '{:6d}' + '\u2502{:7.3f}' * 2 + '\u2502{:6.2f}'
# print(header)

# EPOCHES = 120
# best_loss = 10
# for epoch in range(1, EPOCHES + 1):
#     losses = []
#     start_time = time.time()
#     model.train()
#     for step, data in tqdm(enumerate(train_dataloader), total = len(train_dataloader)):
#         image, target = data["image"].to(device), data["mask"].float().to(device)
#         image = torch.unsqueeze(image, 1)
#         target = torch.unsqueeze(target, 1)
#         optimizer.zero_grad()
#         output = model(image)
#         loss = loss_fn(output, target)
#         loss.backward()
#         optimizer.step()
#         losses.append(loss.item())
#         # print(loss.item())
#
#     vloss = validation(model, test_dataloader, loss_fn)
#     print(raw_line.format(epoch, np.array(losses).mean(), vloss,
#                           (time.time() - start_time) / 60 ** 1))
#     losses = []
#
#     if vloss < best_loss:
#         best_loss = vloss
#         torch.save(model.state_dict(), 'model_best_3d_densenet3d.pth')
