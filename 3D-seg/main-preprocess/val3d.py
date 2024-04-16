import numpy as np
import pandas as pd
import pathlib, sys, os, random, time
import cv2, gc
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryJaccardIndex
from tqdm import tqdm

import matplotlib.pyplot as plt

import warnings

# from preprocess.final import device, loss_fn

warnings.filterwarnings('ignore')

from tqdm import tqdm

import albumentations as A

import rasterio
from rasterio.windows import Window
import monai
from monai.networks.nets import UNet


from final import *
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)
state_dict = torch.load("model_best_unet.pth")

model.load_state_dict(state_dict)
train_raw = load_volume(path= "../data/train/masks/**.jpg")
test_raw = load_volume(path= "../data/test/masks/**.jpg")
train_img, train_mask = train_raw["image"], train_raw["mask"]
test_img, test_mask = test_raw["image"], test_raw["mask"]
# for i in range()
train_img, train_mask = random_crop_3d(train_img, train_mask, [128, 128, 128], train=True)
test_img, test_mask = random_crop_3d(test_img, test_mask, [128, 128, 128], train=True)
# print(np.array(test_img).shape)
temp1= list(zip(train_img, train_mask))
temp2 = list(zip(test_img, test_mask))
random.shuffle(temp1)
random.shuffle(temp2)
res1, res2 = zip(*temp1)
res3, res4 = zip(*temp2)
trainSet1, trainSetGround1 = list(res1), list(res2)
testSet1, testSetGround1 = list(res3), list(res4)

# print(np.array(trainSet).shape)
train_dataset1 = EleMic1(trainSet1, trainSetGround1)
test_dataset1 = EleMic1(testSet1, testSetGround1)
train_loader1 = DataLoader(train_dataset1, batch_size=1, shuffle=True, drop_last= True)
test_loader1 = DataLoader(test_dataset1, batch_size= 1, shuffle= True, drop_last=True)
bar = tqdm(enumerate(test_loader1), total = len(test_loader1))
nums = 0
model.to(device)
model.eval()
raw_line = '\u2502{:7.3f}' + '\u2502{:6.2f}' * 2

def method4(inputs, target):

    intersection = (inputs * target).sum()
    union = inputs.sum() + target.sum()
    dice = (2. * intersection) / (union + 1e-8)


    return dice


metric = BinaryJaccardIndex()
losses = []
dices = []
ious = []
num_correct = 0.0
num_pixels = 0.0
dice_score = 0.0
with torch.no_grad():
    for step, data in bar:
        # with torch.no_grad:
        image, mask = data["image"].to(device), data["mask"].float().to(device)
        image = torch.unsqueeze(image, 1)
        mask = torch.unsqueeze(mask, 1)
        output = model(image)
        # print(f"output is {output.shape}")
        # print(f"mask is  {mask.shape}")
        # if (nums <= 15):
        #     plt.figure(figsize=(16, 8))
        #     plt.subplot(131)
        #     plt.imshow(mask[0][0].cpu(), cmap='gray')
        #     plt.title("original mask")
        #     plt.subplot(132)
        #     plt.imshow(image[0][0].cpu(), cmap='bone')
        #     plt.title("original image")
        #     plt.subplot(133)
        #     # plt.imshow((nn.Sigmoid()(output[0][0])).cpu().detach().float(), cmap='gray')
        #     # (nn.Sigmoid()(outputs[0][0])).cpu().float()
        #     plt.imshow((nn.Sigmoid()(output[0][0])).cpu().float(), cmap='gray')
        #     plt.title("Real output")
        #     plt.show()
        #     nums += 1
        # image = image.to(device)
        # mask = mask.to(device)
        loss = loss_fn(output, mask)
        # print(mask.shape)
        # print(output.shape)
        print(f"output is {output.shape}")
        print(f"mask is  {mask.shape}")
        iou = metric(((nn.Sigmoid()(output).cpu())), mask.cpu())
        output = torch.sigmoid(output)
        output = (output > 0.5).cpu().float()
        # output = nn.Sigmoid()(output).cpu()
        mask = mask.cpu()
        # print(f"output is {output}")
        # print(f"mask is  {mask}")
        print(f"output is {output.shape}")
        print(f"mask is  {mask.shape}")
        dice_score = method4(output, mask)
        num_correct += (output == mask).sum()
        num_pixels += torch.numel(output)
        # dice_score += (2 * (output * mask).sum()) / (2 * (output * mask).sum()+ ((mask * output) < 1).sum())
        # print(dice)
        # print(image.shape)
        # print(mask.shape)
        # print(output.shape)
        losses.append(loss.item())
        dices.append(dice_score.item())
        ious.append(iou)
        # if(nums == 1):
        #     break
    print(f"Got {num_correct/num_pixels} with accuracy {num_correct/num_pixels*100:.2f}")
    # print(f"Dice score: {}")
    print(raw_line.format(np.array(losses).mean(),
                              np.array(dices).mean(), np.array(ious).mean()))