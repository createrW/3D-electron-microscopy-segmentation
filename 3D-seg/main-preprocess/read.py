import SimpleITK as sitk
import numpy as np
from scipy import ndimage
import torch
import random
# img = sitk.ReadImage("../data/te/img/testing.tif")
# img = sitk.GetArrayFromImage(img)
MIN_BOUND = -1000.0
MAX_BOUND = 400.0
def sitk_read_raw(img_path, resize_scale=1): # 读取3D图像并resale（因为一般医学图像并不是标准的[1,1,1]scale）
    nda = sitk.ReadImage(img_path)
    if nda is None:
        raise TypeError("input img is None!!!")
    nda = sitk.GetArrayFromImage(nda)  # channel first
    nda=ndimage.zoom(nda,[resize_scale,resize_scale,resize_scale],order=0) #rescale

    return nda
# target one-hot编码
def to_one_hot_3d(tensor, n_classes=3):  # shape = [batch, s, h, w]
    n, s, h, w = tensor.size()
    one_hot = torch.zeros(n, n_classes, s, h, w).scatter_(1, tensor.view(n, 1, s, h, w), 1)
    return one_hot
def norm_img(image): # 归一化像素值到（0，1）之间，且将溢出值取边界值
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

def random_crop_3d(img, label, crop_size):
    random_x_max = img.shape[0] - crop_size[0]
    random_y_max = img.shape[1] - crop_size[1]
    random_z_max = img.shape[2] - crop_size[2]

    if random_x_max < 0 or random_y_max < 0 or random_z_max < 0:
        return None

    x_random = random.randint(0, random_x_max)
    y_random = random.randint(0, random_y_max)
    z_random = random.randint(0, random_z_max)

    crop_img = img[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1], z_random:z_random + crop_size[2]]
    crop_label = label[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                 z_random:z_random + crop_size[2]]

    return crop_img, crop_label
# from utils.common import *

from scipy import ndimage
import numpy as np
from torchvision import transforms as T
import torch,os
from torch.utils.data import Dataset, DataLoader


class Lits_DataSet(Dataset):
    def __init__(self, crop_size,resize_scale, dataset_path,mode=None):
        self.crop_size = crop_size
        self.resize_scale=resize_scale
        self.dataset_path = dataset_path
        self.n_labels = 3

        # if mode=='train':
        #     self.filename_list = load_file_name_list(os.path.join(dataset_path, 'train_name_list.txt'))
        # elif mode =='val':
        #     self.filename_list = load_file_name_list(os.path.join(dataset_path, 'val_name_list.txt'))
        # else:
        #     raise TypeError('Dataset mode error!!! ')


    def __getitem__(self, index):
        data, target = self.get_train_batch_by_index(crop_size=self.crop_size, index=index,
                                                     resize_scale=self.resize_scale)
        return torch.from_numpy(data), torch.from_numpy(target)

    def __len__(self):
        return len(self.filename_list)

    def get_train_batch_by_index(self,crop_size, index,resize_scale=1):
        img, label = self.get_np_data_3d(self.filename_list[index],resize_scale=resize_scale)
        img, label = random_crop_3d(img, label, crop_size)
        return np.expand_dims(img,axis=0), label

    def get_np_data_3d(self, filename, resize_scale=1):
        data_np = sitk_read_raw(self.dataset_path + '/data/' + filename,
                                resize_scale=resize_scale)
        data_np=norm_img(data_np)
        label_np = sitk_read_raw(self.dataset_path + '/label/' + filename.replace('volume', 'segmentation'),
                                 resize_scale=resize_scale)
        return data_np, label_np

# 测试代码
import matplotlib.pyplot as plt
def main():
    fixd_path  = r'E:\Files\pycharm\MIS\3DUnet\fixed_data'
    dataset = Lits_DataSet([16, 64, 64],0.5,fixd_path,mode='train')  #batch size
    data_loader=DataLoader(dataset=dataset,batch_size=2,num_workers=1, shuffle=True)
    for batch_idx, (data, target) in enumerate(data_loader):
        target = to_one_hot_3d(target.long())
        print(data.shape, target.shape)
        plt.subplot(121)
        plt.imshow(data[0, 0, 0])
        plt.subplot(122)
        plt.imshow(target[0, 1, 0])
        plt.show()
if __name__ == '__main__':
    img = sitk_read_raw("../data/tr/img/training.tif", 0.5)
    mask = sitk_read_raw("../data/tr/mask/training_groundtruth.tif", 0.5)
    print(np.array(img).shape)
    data_np = norm_img(img)
    img, label = random_crop_3d(img, mask, [16, 64, 64])
    img = np.expand_dims(img,axis=0)
    print(img.shape)
    print(label.shape)



