import tifffile as tiff
train_images = tiff.imread('../data/training.tif')
train_masked_images = tiff.imread('../data/training_groundtruth.tif')

test_images = tiff.imread('../data/testing.tif')
test_masked_images = tiff.imread('../data/testing_groundtruth.tif')

val_images = tiff.imread('../data/volumedata.tif')


import os
if not os.path.isdir("../data/train/"):
    os.makedirs("../data/train/")
    if not os.path.isdir("../data/train/images"):
        os.makedirs("../data/train/images")
    if not os.path.isdir("../data/train/masks"):
        os.makedirs("../data/train/masks")

if not os.path.isdir("../data/test/"):
    os.makedirs("../data/test/")
    if not os.path.isdir("../data/test/images"):
        os.makedirs("../data/test/images")
    if not os.path.isdir("../data/test/masks"):
        os.makedirs("../data/test/masks")

if not os.path.isdir("../data/val/"):
    os.makedirs("../data/val/")
    if not os.path.isdir("../data/val/images"):
        os.makedirs("../data/val/images")



import cv2
import shortuuid
import numpy as np
from glob import glob
#for uniquename
s=[]
for i in range(165):
  s.append(str(i))

v = []
for i in range(1065):
    v.append(str(i))
#train

#images:
for img in range(train_images.shape[0]):
    cv2.imwrite("../data/train/images/{}.jpg".format(s[img]),train_images[img])
#masks:
for img in range(train_masked_images.shape[0]):
    cv2.imwrite("../data/train/masks/{}.jpg".format(s[img]),train_masked_images[img])

#test

#images:
for img in range(test_images.shape[0]):
    cv2.imwrite("../data/test/images/{}.jpg".format(s[img]),test_images[img])
#masks:
for img in range(test_masked_images.shape[0]):
    cv2.imwrite("../data/test/masks/{}.jpg".format(s[img]),test_masked_images[img])

#val
for img in range(val_images.shape[0]):
    cv2.imwrite("../data/val/images/{}.jpg".format(v[img]), val_images[img])

print("--------------------------------------------------")
print("Train")
print("No Images:",len(os.listdir("../data/train/images")))
print("No masks:",len(os.listdir("../data/train/masks")))

print("--------------------------------------------------")
print("Test")
print("No Images:",len(os.listdir("../data/test/images")))
print("No masks:",len(os.listdir("../data/test/masks")))

print("--------------------------------------------------")
print("Val")
print("No Images:",len(os.listdir("../data/val/images")))