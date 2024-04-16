import random

import cv2
import numpy as np

train_path ='../data/tr/img/training.tif'
train_gt_path = '../data/tr/mask/training_groundtruth.tif'
test_path = '../data/te/img/testing.tif'
test_gt_path = '../data/te/mask/testing_groundtruth.tif'
import tifffile

# def crop_3d(crop_shape, tif_image):
#     # Calculate the number of 3D slices per dimension
#     num_slices_x = tif_image.shape[2] // crop_shape[0]
#     num_slices_y = tif_image.shape[1] // crop_shape[1]
#     num_slices_z = tif_image.shape[0] // crop_shape[2]
#
#     # Initialize an empty array to store the 3D slices
#     slices_3d = np.zeros((num_slices_x * num_slices_y * num_slices_z, crop_shape[0], crop_shape[1], crop_shape[2]), dtype=np.uint16)
#
#     # Fill the array with the 3D slices
#     idx = 0
#     for z in range(num_slices_z):
#         for y in range(num_slices_y):
#             for x in range(num_slices_x):
#                 start_x = x * crop_shape[0]
#                 end_x = start_x + crop_shape[0]
#                 start_y = y * crop_shape[1]
#                 end_y = start_y + crop_shape[1]
#                 start_z = z * crop_shape[2]
#                 end_z = start_z + crop_shape[2]
#                 slice_3d = tif_image[start_z:end_z, start_y:end_y, start_x:end_x]
#                 # print(f"Slice shape for ({start_x}:{end_x}, {start_y}:{end_y}, {start_z}:{end_z}): {slice_3d.shape}")
#                 if slice_3d.shape == crop_shape:
#                     # print(f"Slice shape for ({start_x}:{end_x}, {start_y}:{end_y}, {start_z}:{end_z}): {slice_3d.shape}")
#                     slices_3d[idx] = slice_3d
#                     idx += 1
#     return slices_3d
# from skimage import io
# tif_image = tifffile.imread(train_path)
# tif_fk = tifffile.imread(train_gt_path)
# crop_shape = (128, 128, 128)
# slices_1 = crop_3d(crop_shape, tif_image)
# slices_2 = crop_3d(crop_shape, tif_fk)
# # Convert the list of slices into a 4D array
# # slices_3d = np.array(slices_3d)
# print(len(slices_1))
# #show
# i = random.randint(0, len(slices_1))
# image = slices_1[i]
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# # print(image[64, :, :].shape)
# axes[0].imshow(image[64, :, :], cmap='bone')
# axes[0].set_title('YZ Plane')
#
# axes[1].imshow(image[:, 64, :], cmap='bone')
# axes[1].set_title('XZ Plane')
#
# axes[2].imshow(image[:, :, 64], cmap='bone')
# axes[2].set_title('XY Plane')
#
# plt.tight_layout()
# plt.show()
#
# mask = slices_2[i]
# print(mask.shape)
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#
# axes[0].imshow(mask[32, :, :], cmap='gray')
# axes[0].set_title('YZ Plane')
#
# axes[1].imshow(mask[:, 32, :], cmap='gray')
# axes[1].set_title('XZ Plane')
#
# axes[2].imshow(mask[:, :, 32], cmap='gray')
# axes[2].set_title('XY Plane')
# plt.show()
img = tifffile.imread(test_path)
print(np.min(img))
print(np.max(img))

