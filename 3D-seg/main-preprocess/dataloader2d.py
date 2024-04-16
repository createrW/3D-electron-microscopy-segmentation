import numpy as np
import random
import matplotlib.pyplot as plt
import copy
from os import path

from albumentations.pytorch import ToTensorV2
from monai.metrics.utils import create_table_neighbour_code_to_surface_area
from torchmetrics.classification import BinaryJaccardIndex
import random
from torch.utils.tensorboard import SummaryWriter


#Libraries for data preprocessing
import torch
from torchvision import transforms
import torchvision

#Libraries for Model building
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
import albumentations as A

#Libraries for loading dataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import PIL
import os

device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()

#Loading the images (ground truth and test/train) 
testRaw=Image.open("../data/te/img/testing.tif")
trainRaw=Image.open("../data/tr/img/training.tif")
trainGroundRaw=Image.open("../data/tr/mask/training_groundtruth.tif")
testGroundRaw=Image.open("../data/te/mask/testing_groundtruth.tif")

#Converting between .tiff file images to tensors which have more functionality
def convertToList(image):
  listImage=[]
  for i in range(image.n_frames):
    image.seek(i)
    imageFrame=torch.squeeze(TF.to_tensor(image))
    listImage.append(imageFrame)
    # print(imageFrame)
    # print(imageFrame)
  return listImage


#Converting all the data to tensors
train=convertToList(trainRaw)
trainGround=convertToList(trainGroundRaw)
test=convertToList(testRaw)
testGround=convertToList(testGroundRaw)


#2d -> 3d whatever u want
def extractSubVolume(image, dimension):
  length=dimension[1]
  width=dimension[2]
  depth=dimension[0]
  # print(f"{depth}:{width}:{length}")
  voxel=image[depth][length[0]:length[1],width[0]:width[1]]
  # print("")
  # print(voxel.shape)
  # print(voxel[0])
  # print(voxel[1])
  # print(voxel.shape)
  return voxel


def extractVoxels(image, linspacing):
    dLength = linspacing[0]
    dWidth = linspacing[1]

    sDepth = len(image)
    sLength = len(image[0])
    sWidth = len(image[0][0])

    image_set = []
    for depth in range(0, sDepth):
        for length in range(0, sLength, dLength):
            for width in range(0, sWidth, dWidth):
                if (length + dLength < sLength and width + dWidth < sWidth):
                    # print(depth, length, width)
                    dimension = [depth, [length, length + dLength], [width, width + dWidth]]
                    # print(dimension)
                    image_set.append(extractSubVolume(image, dimension))
    return image_set


#Extracting the input to the model as well as the model output in terms of subvolumes of size represented by dimensions
dimension=[512,512]
trainPre=extractVoxels(train, dimension)
print(np.array(trainPre).shape)
trainGroundPre=extractVoxels(trainGround, dimension)
testPre=extractVoxels(test,dimension)
testSetGroundPre=extractVoxels(testGround, dimension)
# print(trainPre)
# print(train[0])
# print(train[1])
# print(train[2])
# i = train[1]
# j = train[2]
# print(len(train))
# print(len(train[0]))
# print(len(train[0][0]))
# print(train[0][i[0]:i[1], j[0]:j[2]])
#Dataset class to make the code modular along with transform functionality
class EleMic(Dataset):
  def __init__(self,image, target, transforms=None):
    self.transforms=transforms
    self.target=target
    self.image=image

  def __len__(self):
    return(len(self.image))

  def __getitem__(self, idx):
    image=self.image[idx]
    if self.transforms:
      image=self.transforms(image)
    target=self.target[idx]
    # return (image.to(device), target.to(device))
    return {
        'image': image.to(device),
        'mask': target.to(device)
    }



# i=random.randrange(0,len(trainPre))
# plt.imshow(trainPre[i], cmap='gray')
#
#Augmenting the data with more training inputs via abberations in the original image
def augmentDataset(set_train, set_ground, threshold, iter, train = False):
  final_train_data=copy.copy(set_train)
  final_set_ground=copy.copy(set_ground)
  i=0
  for image in set_train:

    if(image.count_nonzero()>threshold):
      final_train_data.append(TF.hflip(image))
      final_set_ground.append(TF.hflip(set_ground[i]))
      final_train_data.append(TF.vflip(image))
      final_set_ground.append(TF.vflip(set_ground[i]))
      ###
      final_train_data.append(TF.adjust_brightness(image,2.0))
      final_set_ground.append(TF.adjust_brightness(set_ground[i], 2.0))
      # final_train_data.append(TF.adjust_sharpness(image, 2.0))
      # final_set_ground.append(TF.adjust_sharpness(set_ground[i], 2.0))
      # transformed_2 = A.ShiftScaleRotate(p=0.5)
      # transform = A.Compose([
      #     A.ShiftScaleRotate(shift_limit=0.3,
      #                        scale_limit=(-0.45, 0.05),
      #                        rotate_limit=45,
      #                        # value=0,
      #                        border_mode=4,
      #                        p=0.95),
      #     A.RandomBrightnessContrast(p=1.0),
      #     A.RandomGamma(p=0.8),
      # ])
      if train:
          transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.RandomBrightnessContrast(brightness_limit=0.1, p=0.7),
            # A.OneOf([
            #         A.GaussNoise(var_limit = (10.0, 50.0)),
            #         A.GaussianBlur(),
            #         A.MotionBlur(),
            #         A.MedianBlur(blur_limit=3),
            #         ], p=0.4),
            # A.OneOf([
            #     A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            #     A.OpticalDistortion(distort_limit=1., p=1.0)
            # ],p=0.2),
            # A.ShiftScaleRotate(p=0.7, scale_limit=0.5, shift_limit=0.2, rotate_limit=30),
            # A.CoarseDropout(max_holes=1, max_height=0.25, max_width=0.25),
            # ToTensorV2(transpose_mask=True)
        ])
          transform = transform(image=np.array(image), mask = np.array(set_ground[i]))
          # transform = transform(image=image, mask=set_ground[i])
          # print(image.dtype)
          # print(image)
          # print(set_ground[i].dtype)

          transformed_image = torch.tensor(transform['image'])
          transformed_mask = torch.tensor(transform['mask'])
          # print(transformed_mask.dtype)
          final_train_data.append(transformed_image)
          final_set_ground.append(transformed_mask)
      final_train_data.append(torch.randn(image.shape)+image)
      final_set_ground.append(set_ground[i])
    i+=1
  return final_train_data, final_set_ground

#
#
trainPre, trainGroundPre = augmentDataset(trainPre, trainGroundPre, threshold=0, iter=1, train=True)
temp = list(zip(trainPre, trainGroundPre))
random.shuffle(temp)
res1, res2 = zip(*temp)
# print(res1)
# # res1 and res2 come out as tuples, and so must be converted to lists.
trainSet, trainSetGround = list(res1), list(res2)
print(np.asarray(trainSet).shape)
testPre, testGroundPre = augmentDataset(testPre, testSetGroundPre, threshold = 0, iter = 1)
temp = list(zip(testPre, testGroundPre))
random.shuffle(temp)
res1, res2 = zip(*temp)
# res1 and res2 come out as tuples, and so must be converted to lists.
testpre, testgroundPre = list(res1), list(res2)

valTestSplit=0.8
iter=1
valSet=testpre[(int)(len(testpre)*valTestSplit):]
valSetGround=testgroundPre[(int)(len(testgroundPre)*valTestSplit):]

testSetGround=testgroundPre[:(int)(len(testgroundPre)*valTestSplit)]
testSet=testpre[:(int)(len(testpre)*valTestSplit)]

dataTransforms = {
    'train': None,
    'test': None
    , 'val': None
}

dataSets = {
    'train': trainSet,
    'test': testSet
    , 'val': valSet
}
ground_cubes = {
    'train': trainSetGround,
    'test': testSetGround
    , 'val': valSetGround

}

batchSize = 1

#Loading the datasets and the data loaders
dataSets= {x: EleMic(dataSets[x], ground_cubes[x], dataTransforms[x])
            for x in ['train', 'test', 'val']}

dataset_sizes = {x: len(dataSets[x]) for x in ['train', 'test','val']}

dataLoaders={x:torch.utils.data.DataLoader(dataSets[x], batch_size=batchSize,shuffle=True,drop_last=True)
              for x in ['train', 'test','val']}

# print(dataset_sizes)
i = random.randint(0,len(dataSets["train"]))
data = dataSets["train"][i]
img = data["image"].cpu().numpy()
mask = data["mask"].cpu().numpy()
print(img.shape)
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.subplot(122)
plt.imshow(mask, cmap = 'gray')
plt.show()

#
# #change model
# import sys, os
# import cv2
# import pandas as pd
# from glob import glob
# import numpy as np
#
# from timeit import default_timer as timer
# # !python -m pip install --no-index --find-links=/kaggle/input/pip-download-for-segmentation-models-pytorch segmentation-models-pytorch
# import segmentation_models_pytorch as smp
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
from torch.cuda.amp import autocast
import matplotlib
import matplotlib.pyplot as plt
# model = smp.Unet(
#     encoder_name="resnet34",
#     encoder_weights="imagenet",
#     in_channels=1,
#     classes=1,
# )
# from models.resnet26d import *
# model = SenUNetStem(
#             encoder_name="maxvit_tiny_tf_512.in1k",
#             classes=1,
#             activation=None,
#         )


# class DiceBCELoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(DiceBCELoss, self).__init__()
#
#     def forward(self, inputs, targets, smooth=1):
#         # comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = nn.Sigmoid()(inputs)
#
#         # flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
#
#         intersection = (inputs * targets).sum()
#         dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
#         BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
#         Dice_BCE = BCE + dice_loss
#
#         return Dice_BCE
#
#
# loss = DiceBCELoss()
#
#
# # loss = CustomLoss()
# learningRate=1e-4
# model.to(device) #migrating the model to cuda compatible if GPU server exists
# optimizer=optim.Adam(model.parameters(), lr=learningRate, weight_decay=0.01)
#
# # training loop
# numEpochs = 70
#
# tlossPerEpoch = []
# vlossPerEpoch = []
# tIndexPerEpoch = []
# vIndexPerEpoch = []
#
# metric = BinaryJaccardIndex()
#
# for epoch in range(numEpochs):
#     epoch_loss = 0
#     model.train()
#     btc = []
#     for idx, batch in enumerate(dataLoaders['train']):
#         optimizer.zero_grad()
#         # clearing out the previous gradient calculations
#         # migrating the code to cuda device if GPU servers are present
#         imgInput = batch[0].to(device)
#         imgTarget = batch[1].to(device)
#
#         # Images input into the model
#         imgInput = torch.unsqueeze(imgInput, 1)
#         outputs = model(imgInput)
#         imgTarget = torch.unsqueeze(imgTarget, 1)
#         # print("Target", idx, imgTarget.count_nonzero())
#         batchLoss = loss(outputs, imgTarget)
#
#         # Updating the weights
#         batchLoss.backward()
#         optimizer.step()
#         # scheduler.step()
#
#         # loss updates
#         epoch_loss += batchLoss.detach()
#         btc.append(metric(((nn.Sigmoid()(outputs).cpu())), imgTarget.cpu()))
#
#     tlossPerEpoch.append(epoch_loss.item() / len(dataLoaders['train']))
#     tIndexPerEpoch.append(np.mean(btc))
#     # writer.add_scalar("Loss/train/", epoch_loss, epoch,np.mean(btc))
#
#     btc = []
#     valLoss = 0
#     with torch.no_grad():
#         # set the model in evaluation mode
#         model.eval()
#         # loop over the validation set
#         for idx, batch in enumerate(dataLoaders['val']):
#             # send the input to the device
#             x = batch[0].to(device)
#             y = batch[1].to(device)
#             # make the predictions and calculate the validation loss
#             x = torch.unsqueeze(x, 1)
#             pred = model(x)
#             y = torch.unsqueeze(y, 1)
#             valLoss += loss(pred, y).detach()
#             btc.append(metric(((nn.Sigmoid()(pred).cpu())), y.cpu()))
#     vlossPerEpoch.append(valLoss.item() / len(dataLoaders['val']))
#     vIndexPerEpoch.append(np.mean(btc))
#
#     # the logging/saving of training attributes and losses
#     writer.add_scalar("Loss/val/", valLoss, epoch, np.mean(btc))
#
#     if (epoch % 5 == 0):
#         # pathToSave = os.path.join(os.getcwd(), 'ResUnet1', 'learningRate' + str(learningRate), str(batchSize),
#         #                           str(epoch))
#         pathToSave = os.path.join("../trained_models/2d", 'resnet34' )
#         if path.exists(pathToSave) == False:
#             os.makedirs(pathToSave)
#         torch.save(model.state_dict(), "" + ".pth")
#         # torch.save(tlossPerEpoch, pathToSave)
#         with torch.no_grad():
#             # set the model in evaluation mode
#             model.eval()
#             print("Intermediate training sample")
#             plt.figure(figsize=(12, 6))
#             plt.subplot(131)
#             plt.imshow((nn.Sigmoid()(outputs[0][0])).cpu().float(), cmap='gray')
#             plt.title("Model Output")
#             plt.subplot(132)
#             plt.imshow(imgInput[0][0].cpu(), cmap='gray')
#             plt.title("Input")
#             plt.subplot(133)
#             plt.imshow(imgTarget[0][0].cpu(), cmap='gray')
#             plt.title("Real Output")
#             plt.show()
#
#             print("Intermediate validation sample")
#             plt.figure(figsize=(12, 6))
#             plt.subplot(131)
#             plt.imshow((nn.Sigmoid()(pred[0][0])).cpu().float(), cmap='gray')
#             plt.subplot(132)
#             plt.imshow(x[0][0].cpu(), cmap='gray')
#             plt.subplot(133)
#             plt.imshow(y[0][0].cpu(), cmap='gray')
#             plt.show()
#     # printing the epochs
#     print("Epoch %2i : Train Loss %f with Jaccard Index/IoU %f, Val loss %f with Jaccard Index/IoU %f" % (
#         epoch, tlossPerEpoch[-1], np.mean(tIndexPerEpoch[-1]), vlossPerEpoch[-1], np.mean(vIndexPerEpoch[-1])))
# writer.flush()