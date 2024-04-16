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

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
import torchvision
import torchmetrics
# Utils
import joblib
from tqdm import tqdm
from collections import defaultdict

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

# For descriptive error messages
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
checkpoint_path =  '../trained_models/timm-resnest14d/timm-resnest14d_ep46_loss0.234_IOU0.742.pt'

CONFIG = {
    "seed": 42,
    "epochs": 50,
    "img_size": 1024,
    "model_name": "timm-regnetx_016",
    "checkpoint_path" : checkpoint_path,
    "pretrained" : None,
    "num_classes": 1,
    "train_batch_size": 12,
    "valid_batch_size": 12,
    "learning_rate": 1e-4,
    "scheduler": 'CosineAnnealingLR',
    "min_lr": 1e-7,
    "T_max": 500,
    "weight_decay": 0.01,
    "fold" : 1,
    "n_fold": 5,
    "n_accumulate": 1,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}
SAVE = '../trained_models/'#保存模型路径
SAVE = SAVE + CONFIG["model_name"]
if not os.path.isdir(SAVE):
    os.makedirs(SAVE)

print(f"save to {SAVE}")

print(CONFIG['device'])
def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(CONFIG['seed'])

from preprocess.dataloader2d import *
# from Others.unet import *
import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
# model = smp.Unet(
#         encoder_name='se_resnext101_32x4d',
#         encoder_weights="imagenet",
#         in_channels=1,
#         classes=1,
#         activation=None,
#     )
from Others.unet import *
model = UNet(1,1)
# state_dict = torch.load(CONFIG['checkpoint_path'])
#
# model.load_state_dict(state_dict)
print(torch.cuda.is_available())
print(f"using model {CONFIG['model_name']}")
#Loss
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = nn.Sigmoid()(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


def fetch_scheduler(optimizer):
    if CONFIG['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['T_max'],
                                                   eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3,
                                                             eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == None:
        return None

    return scheduler


from BoundaryDoULoss.utils import *
# Loss = torch.nn.BCEWithLogitsLoss()
# # Loss = BoundaryDoULoss(1)
# scaler = torch.cuda.amp.GradScaler()
# optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'],
#                        weight_decay=CONFIG['weight_decay'])
# scheduler = fetch_scheduler(optimizer)
# from monai.metrics import DiceMetric, HausdorffDistanceMetric
# metric = BinaryJaccardIndex()
# metric = DiceMetric(include_background=False,  # 原此处为True
#                                   reduction='mean_batch',
#
#                                   get_not_nans=False)
model = model.to(CONFIG['device'])


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
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
def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=1e-4, weight_decay=1e-3)
    losses = []
    start_time = time.time()
    model.train()
    for image, target in tqdm(dataloader):
        image, target = image.to(CONFIG["device"]), target.float().to(CONFIG["device"])
        optimizer.zero_grad()
        output = model(image)['out']
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        # print(loss.item())

    vloss = validation(model, dataLoaders['train'], loss_fn)
    print(raw_line.format(epoch, np.array(losses).mean(), vloss,
                          (time.time() - start_time) / 60 ** 1))
    losses = []


@torch.no_grad()
def validation(model, loader, loss_fn):
    losses = []
    model.eval()
    for image, target in loader:
        image, target = image.to(CONFIG["device"]), target.float().to(CONFIG["device"])
        output = model(image)['out']
        loss = loss_fn(output, target)
        losses.append(loss.item())

    return np.array(losses).mean()

def run_training(model, optimizer, scheduler, device, num_epochs):
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_iou = -np.inf
    best_epoch_loss = np.inf
    history = defaultdict(list)
    header = r'''
            Train | Valid
    Epoch |  Loss |  Loss | Time, m
    '''
    #          Epoch         metrics            time
    raw_line = '{:6d}' + '\u2502{:7.3f}' * 2 + '\u2502{:6.2f}'
    print(header)

    for epoch in range(1, num_epochs + 1):
        flag = 0
        gc.collect()
        print(np.array(dataLoaders['train']).shape)
        train_epoch_loss, train_epoch_iou = train_one_epoch(model, optimizer, scheduler,
                                                            dataloader=dataLoaders['train'],
                                                            device=CONFIG['device'], epoch=epoch)

        val_epoch_loss, val_epoch_iou = valid_one_epoch(model, dataLoaders['test'], device=CONFIG['device'],
                                                                          epoch=epoch)

        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        history['Train IOU'].append(train_epoch_iou)
        history['Valid IOU'].append(val_epoch_iou)
        # history['Train Recall'].append(train_epoch_recall)
        # history['Valid Recall'].append(val_epoch_recall)
        history['lr'].append(scheduler.get_lr()[0])

        # deep copy the model
        if best_epoch_loss >= val_epoch_loss:
            flag = 1
            print(f"Validation loss Improved ({best_epoch_loss} ---> {val_epoch_loss})")
            best_epoch_loss = val_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"{SAVE}/{CONFIG['model_name']}_ep{epoch}_loss{best_epoch_loss:.3f}_IOU{val_epoch_iou:.3f}.pt"
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved.")

        elif epoch >= num_epochs - 8:
            flag = 1
            PATH = f"{SAVE}/{CONFIG['model_name']}_ep{epoch}_loss{val_epoch_loss:.3f}.pt"
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved.")
        if best_epoch_iou <= val_epoch_iou:
             print(f"Jaccard Index/IoU Improved ({best_epoch_iou} ---> {val_epoch_iou})")
             best_epoch_iou = val_epoch_iou
             best_model_wts = copy.deepcopy(model.state_dict())
             PATH = f"{SAVE}/{CONFIG['model_name']}_ep{epoch}_loss{val_epoch_loss:.3f}_IOU{best_epoch_iou:.3f}.pt"
             if flag == 0:
                torch.save(model.state_dict(), PATH)
                # Save a model file from the current directory
                print(f"Model Saved.")
             else:
                 print("Model already saved.")

        if(epoch % 5 ==0 ):
            test_loss, test_iou = valid_one_epoch(model, dataLoaders['val'],device=CONFIG['device'],
                                                                          epoch=epoch)
            print(f"The True loss is {test_loss}, IOU is {test_iou}")



    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best loss: {:.4f}".format(best_epoch_loss))
    print("Best Jaccard Index/IoU: {:.4f}".format(best_epoch_iou))

    return model, history


if __name__ == "__main__":
    for epoch in (1, 10):
        train_one_epoch(model, None, None,
                    dataloader=dataLoaders['train'],
                    device=CONFIG['device'], epoch=epoch)

    # model, history = run_training(model, optimizer, scheduler,
    #                               device=CONFIG['device'],
    #                               num_epochs=CONFIG['epochs'])

    # print(history)