import gc
import cv2
import math
import copy
import time
import random
import glob
from matplotlib import pyplot as plt
import os
# For data manipulation
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from Vnet3D.net3 import *
from preprocess.dataset3d import *
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
checkpoint_path =  '../Vnet3D/UNET3D_ep6_loss0.858_IOU0.000.pt'

CONFIG = {
    "seed": 42,
    "epochs": 50,
    "img_size": 1024,
    "model_name": "unet3dbasic",
    "checkpoint_path" : checkpoint_path,
    "pretrained" : None,
    "num_classes": 5,
    "train_batch_size": 12,
    "valid_batch_size": 12,
    "learning_rate": 5e-5,
    "scheduler": 'None',
    "min_lr": 1e-7,
    "T_max": 500,
    "weight_decay": 1e-6,
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

from preprocess.dataloader3d  import *
from Others.unet import *
# model = smp.Unet(
#         encoder_name=CONFIG["model_name"],
#         encoder_weights=None,
#         in_channels=1,
#         classes=1,
#         activation=None,
#     )
#
from Vnet3D.layer import *
from my_3D.unet3dw import *
model = UNet(in_dim=1, out_dim=1, num_filters=4)
# model = Basic3DUNet()
# state_dict = torch.load(CONFIG['checkpoint_path'])
#
# model.load_state_dict(state_dict)
model.to(CONFIG['device'])
print(torch.cuda.is_available())
print(f"using model {CONFIG['model_name']}")
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
        # BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = dice_loss

        return Dice_BCE


def fetch_scheduler(optimizer):
    scheduler1 = ''
    if CONFIG['scheduler'] == 'CosineAnnealingLR':
        scheduler1 = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['T_max'],
                                                   eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler1 = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3,
                                                             eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == None:
        return None

    return scheduler1


from BoundaryDoULoss.utils import *
Loss = DiceBCELoss()
# Loss = BoundaryDoULoss(1)
scaler = torch.cuda.amp.GradScaler()
model = UNet(in_dim=1, out_dim=1, num_filters=4).to(CONFIG["device"])
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'],
                       weight_decay=CONFIG['weight_decay'])
scheduler = fetch_scheduler(optimizer)
metric = BinaryJaccardIndex()
def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    epoch_IOU = 0.0

    dataset_size = 0
    running_loss = 0.0
    running_acc = 0.0
    running_recall = 0.0

    running_IOU = 0.0
    running_dice = 0.0
    epoch_dice = 0.0
    btc = []

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        optimizer.zero_grad()
        images = data['volume'].to(device)
        masks = data['target'].to(device)
        batch_size = images.size(0)
        Iou = 0.0
        outputs = model(images)
        outputs = torch.squeeze(outputs, 1)
        masks = torch.squeeze(masks, 1)
        outputs = torch.squeeze(outputs, 1)
        masks = torch.squeeze(masks, 1)
        #         images = torch.unsqueeze(images, 1)
        #         masks = torch.unsqueeze(masks, 1)
        #         print(outputs)
        #         print("...")
        #         print(masks)
        loss = Loss(outputs, masks)
        loss = loss / CONFIG['n_accumulate']
        dice = metric(((nn.Sigmoid()(outputs).cpu())), masks.cpu())

        # loss.backward()
        scaler.scale(loss).backward()

        # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if (step + 1) % CONFIG['n_accumulate'] == 0:
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            # zero the parameter gradients
            # optimizer.zero_grad()
            optimizer.step()

            # if scheduler is not None:
            #     scheduler.step()

        # _, predicted = torch.max(torch.nn.Softmax(dim=1)(outputs), 1)
        # acc = torch.sum(predicted == masks)
        # # recall_nn = torchmetrics.Recall(task="binary", average='macro', num_classes=CONFIG["num_classes"]).cuda()
        # # recall = recall_nn(predicted, masks)
        #
        # running_loss += (loss.item() * batch_size)
        # running_acc += acc.item()
        # # running_recall += (recall.item() * batch_size)
        # dataset_size += batch_size
        #
        # epoch_loss = running_loss / dataset_size
        # epoch_acc = running_acc / dataset_size
        # epoch_recall = running_recall / dataset_size
        # running_dice += dice.detach().cpu().numpy()
        running_dice += (dice.item() * batch_size)
        print(running_dice)
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        btc.append(metric(((nn.Sigmoid()(outputs).cpu())), masks.cpu()))
        epoch_loss = running_loss / dataset_size
        epoch_dice = running_dice / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'], IOU=epoch_dice, dice=np.mean(btc))
    gc.collect()

    return epoch_loss, Iou


@torch.inference_mode()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()

    epoch_loss = 0.0
    epoch_acc = 0.0
    epoch_IOU = 0.0

    dataset_size = 0
    running_loss = 0.0
    running_acc = 0.0
    running_recall = 0.0

    running_IOU = 0.0
    running_dice = 0.0
    epoch_dice = 0.0
    btc = []

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data['volume'].to(device)
        masks = data['target'].to(device)
        # images = torch.unsqueeze(images, 1)
        # masks = torch.unsqueeze(masks, 1)
        batch_size = images.size(0)
        Iou = 0.0
        outputs = model(images)
        outputs = torch.squeeze(outputs, 1)
        masks = torch.squeeze(masks, 1)
        outputs = torch.squeeze(outputs, 1)
        masks = torch.squeeze(masks, 1)
        #         images = torch.unsqueeze(images, 1)
        #         masks = torch.unsqueeze(masks, 1)
        #         print(outputs)
        #         print("...")
        #         print(masks)
        with amp.autocast():
            loss = Loss(outputs, masks)
            loss = loss / CONFIG['n_accumulate']
            dice = metric(((nn.Sigmoid()(outputs).cpu())), masks.cpu())

        # loss.backward()
        # scaler.scale(loss).backward()

        # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # if (step + 1) % CONFIG['n_accumulate'] == 0:
        #     # optimizer.step()
        #     scaler.step(optimizer)
        #     scaler.update()
        #     # zero the parameter gradients
        #     optimizer.zero_grad()
        #
        #     if scheduler is not None:
        #         scheduler.step()

        # _, predicted = torch.max(torch.nn.Softmax(dim=1)(outputs), 1)
        # acc = torch.sum(predicted == masks)
        # # recall_nn = torchmetrics.Recall(task="binary", average='macro', num_classes=CONFIG["num_classes"]).cuda()
        # # recall = recall_nn(predicted, masks)
        #
        # running_loss += (loss.item() * batch_size)
        # running_acc += acc.item()
        # # running_recall += (recall.item() * batch_size)
        # dataset_size += batch_size
        #
        # epoch_loss = running_loss / dataset_size
        # epoch_acc = running_acc / dataset_size
        # epoch_recall = running_recall / dataset_size
        running_dice += dice.item()
        print(running_dice)
        running_loss += loss.item()
        dataset_size += batch_size
        btc.append(metric(((nn.Sigmoid()(outputs).cpu())), masks.cpu()))
        epoch_loss = running_loss / dataset_size
        epoch_dice = running_dice / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                         IOU=epoch_dice, dice=np.mean(btc))
    gc.collect()

    return epoch_loss, Iou


def run_training(model, optimizer, scheduler, device, num_epochs):
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_iou = -np.inf
    best_epoch_loss = np.inf
    history = defaultdict(list)
    train_dataset = EMdataset3d(mode='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)

    test_dataset = EMdataset3d(mode='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)
    for epoch in range(1, num_epochs + 1):
        flag = 0
        gc.collect()
        train_epoch_loss, train_epoch_iou = train_one_epoch(model, optimizer, scheduler,
                                                            dataloader=train_loader,
                                                            device=CONFIG['device'], epoch=epoch)

        val_epoch_loss, val_epoch_iou = valid_one_epoch(model, test_loader, device=CONFIG['device'],
                                                                          epoch=epoch)

        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        history['Train IOU'].append(train_epoch_iou)
        history['Valid IOU'].append(val_epoch_iou)
        # history['Train Recall'].append(train_epoch_recall)
        # history['Valid Recall'].append(val_epoch_recall)
        # history['lr'].append(scheduler.get_lr()[0])

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

        # if(epoch % 5 ==0 ):
        #     test_loss, test_iou = valid_one_epoch(model, None,device=CONFIG['device'],
        #                                                                   epoch=epoch)
        #     print(f"The True loss is {test_loss}, IOU is {test_iou}")



    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best loss: {:.4f}".format(best_epoch_loss))
    print("Best Jaccard Index/IoU: {:.4f}".format(best_epoch_iou))

    return model, history


def test_1():


    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=1, num_classes=1).to(CONFIG["device"])
    train_dataset = EMdataset3d(mode='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    loss,iou = train_one_epoch(model, optimizer, scheduler, train_loader, device, 1)
    print(loss)
    print(iou)


if __name__ == "__main__":
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # # model = UNet3D(in_channels=1, num_classes=1).to(CONFIG["device"])
    # model = UNet(in_dim=1, out_dim=1, num_filters=4).to(CONFIG["device"])
    # state_dict = torch.load(CONFIG['checkpoint_path'])
    #
    # model.load_state_dict(state_dict)
    # model.to(CONFIG['device'])
    # print(model)
    model, history = run_training(model, optimizer, scheduler,
                                  device=CONFIG['device'],
                                  num_epochs=CONFIG['epochs'])

    # print(history)
    # test_1()
    # train_dataset = EMdataset3d(mode='train')
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)
    #
    # test_dataset = EMdataset3d(mode='test')
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)
    # val_epoch_loss, val_epoch_iou = valid_one_epoch(model, test_loader, device=CONFIG['device'],
    #                                                 epoch=1)