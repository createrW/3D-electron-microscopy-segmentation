import numpy as np
import pandas as pd
import pathlib, sys, os, random, time
import cv2, gc
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm

import albumentations as A

import rasterio
from rasterio.windows import Window
import monai
from monai.networks.nets import UNet


from preprocess.dataloader2d import *

# model = UNet(
#     spatial_dims=2,
#     in_channels=1,
#     out_channels=1,
#     channels=(16, 32, 64, 128, 256),
#     strides=(2, 2, 2, 2),
#     num_res_units=2,
# )
# from Others.unet import *
# model = UNet(1, 1)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
model = smp.Unet(
        encoder_name='resnext101_32x8d',
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    )
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(),
                  lr=1e-4, weight_decay=1e-3)
bar = tqdm(enumerate(dataLoaders["train"]), total = len(dataLoaders["train"]))
for step, data in bar:
    print(data["image"].shape)
    image = data["image"].cpu()
    mask = data["mask"].cpu()
    # image = image.to(device)
    # mask = mask.to(device)
    print(image.shape)
    print(mask.shape)
    # plt.figure(figsize=(16, 8))
    # plt.subplot(121)
    # plt.imshow(mask[0], cmap='gray')
    # plt.subplot(122)
    # plt.imshow(image[0], cmap = 'bone')
    # plt.show()


@torch.no_grad()
def validation(model, loader, loss_fn):
    losses = []
    model.eval()
    bar = tqdm(enumerate(loader), total = len(loader))
    for step, data in bar:
        image, target = data["image"].to(device), data["mask"].float().to(device)
        image = torch.unsqueeze(image, 1)
        target = torch.unsqueeze(target, 1)
        # image, target = image.to(device), target.float().to(device)
        output = model(image)
        loss = loss_fn(output, target)
        losses.append(loss.item())

    return np.array(losses).mean()


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
    return 0.5 * bce + 0.5 * dice


header = r'''
        Train | Valid
Epoch |  Loss |  Loss | Time, m
'''
#          Epoch         metrics            time
raw_line = '{:6d}' + '\u2502{:7.3f}' * 2 + '\u2502{:6.2f}'
print(header)

EPOCHES = 60
best_loss = 10
for epoch in range(1, EPOCHES + 1):
    losses = []
    start_time = time.time()
    model.train()
    for step, data in tqdm(enumerate(dataLoaders["train"]), total = len(dataLoaders["train"])):
        image, target = data["image"].to(device), data["mask"].float().to(device)
        image = torch.unsqueeze(image, 1)
        target = torch.unsqueeze(target, 1)
        optimizer.zero_grad()
        output = model(image)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        # print(loss.item())

    vloss = validation(model, dataLoaders["val"], loss_fn)
    print(raw_line.format(epoch, np.array(losses).mean(), vloss,
                          (time.time() - start_time) / 60 ** 1))
    losses = []

    if vloss < best_loss:
        best_loss = vloss
        torch.save(model.state_dict(), 'model_best_custom_resnext101_32x8d.pth')







