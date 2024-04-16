import copy
import time
from typing import Dict
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import albumentations as A
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F
def preprocess_mask(mask):
    mask = mask.astype(np.float32)
    mask[mask == 0] = 0.0
    mask[mask == 255] = 1.0
    return mask
def random_augmentation(volume, mask):
    # Random rotation (90-degree increments)
    rotation_axes = [(0, 1), (0, 2), (1, 2)]
    axis = np.random.choice([0, 1, 2])
    angle = np.random.choice([0, 90, 180, 270])
    volume = np.rot90(volume, angle // 90, axes=rotation_axes[axis])
    mask = np.rot90(mask, angle // 90, axes=rotation_axes[axis])

    # Random flips
    if np.random.rand() > 0.5:
        volume = np.flip(volume, axis=0)
        mask = np.flip(mask, axis=0)
    if np.random.rand() > 0.5:
        volume = np.flip(volume, axis=1)
        mask = np.flip(mask, axis=1)
    if np.random.rand() > 0.5:
        volume = np.flip(volume, axis=2)
        mask = np.flip(mask, axis=2)

        # if np.random.rand() > 0.5:
        #     original_shape = volume.shape
        #
        #     zoom_factor = 1 + np.random.uniform(-0.2, 0.2)
        #     volume = zoom(
        #         volume, zoom_factor, order=1
        #     )  # Using bilinear interpolation for volume
        #     mask = zoom(
        #         mask, zoom_factor, order=0
        #     )  # Using nearest-neighbor interpolation for mask
        #
        #     # If zooming out, crop to original size. If zooming in, pad with zeros to original size.
        #     volume, mask = self.match_shape(volume, original_shape), self.match_shape(
        #         mask, original_shape
        #     )

        # if np.random.rand() > 0.3:
        #     brightness_factor = np.random.uniform(
        #         0.8, 1.1
        #     )  # Adjust this range as needed
        #     volume = (volume * brightness_factor).astype(np.uint16)
        #     volume = np.clip(volume, 0, 65535)

    return volume, mask
def load_volume(path: str) -> Dict:
    dataset = sorted(glob.glob(path))

    volume = None
    target = None
    # print(dataset)
    for z, path in enumerate(tqdm(dataset)):
        print(f"loading masks ....{path}")
        # mask = (cv2.imread(path, 0) > 127.0) * 1.0
        mask = cv2.imread(path,  cv2.IMREAD_ANYDEPTH)
        mask = preprocess_mask(mask)
        # path = path.replace(
        #     "labels",
        #     "images",
        # ).replace(".png", ".jp2")
        # if "/kidney_3_dense/" in path:
        #     path = path.replace("kidney_3_dense", "kidney_3_sparse")
        path = path.replace("masks", "images")
        # path = path.replace("training_groundtruth", "training")
        # print(path)
        print(f"loading images {path}")
        image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        # img = cv2.resize(image, (256, 256))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = image / 255.
        image = np.array(img, dtype=np.float64)
        if volume is None:
            volume = np.zeros((len(dataset), *image.shape[-2:]), dtype=np.float64)
            target = np.zeros((len(dataset), *mask.shape[-2:]), dtype=np.float64)
        volume[z] = image
        target[z] = mask

    return {"image": volume, "mask": target}

def normilize(image: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
        # xmin = np.min(image)
        # xmax = np.max(image)
        image = (image - xmin) / (xmax - xmin)
        image = np.clip(image, 0, 1)
        return image.astype(np.float32)
def random_crop_3d(img, label, crop_size, train = False):
    random_x_max = img.shape[0] - crop_size[0]
    random_y_max = img.shape[1] - crop_size[1]
    random_z_max = img.shape[2] - crop_size[2]
    final_img = []
    final_mask = []
    num = 0
    if random_x_max < 0 or random_y_max < 0 or random_z_max < 0:
        return None
    for x_random in range(0, random_x_max, crop_size[0]):
        for y_random in range(0, random_y_max, crop_size[0]):
            for z_random in range(0, random_z_max, crop_size[0]):

    # x_random = random.randint(0, random_x_max)
    # y_random = random.randint(0, random_y_max)
    # z_random = random.randint(0, random_z_max)
                if(x_random < random_x_max and y_random < random_y_max and z_random < random_z_max):
                    crop_img = img[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1], z_random:z_random + crop_size[2]]
                    crop_label = label[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                                 z_random:z_random + crop_size[2]]
                    # if(num == 1500):
                    #     return final_img, final_mask
                    #normalize
                    if train:
                        crop_img, crop_label = random_augmentation(crop_img, crop_label)
                        # crop_img = normilize(crop_img, 22, 244)


                    crop_img, crop_label = np.ascontiguousarray(crop_img), np.ascontiguousarray(crop_label)
                    final_img.append(crop_img)
                    final_mask.append(crop_label)
                    num+=1
    # print(num)
    return final_img, final_mask


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
        if self.transforms:
            image = self.transforms(image=image, mask=target)["image"]
            image = image.copy()
            mask = self.transforms(image=image, mask=target)["mask"]
        # return (image.to(device), target.to(device))
        return {
            'image': torch.from_numpy(image),
            'mask': torch.from_numpy(target)
        }
train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, p=0.7),
        ])
val_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)
        ])

# from monai.networks.nets import UNet
from my_3D.unet3d_2 import *
model = UNet(1,1,1)
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

#define metrics
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

loss_fn = UncertaintyEstimationLoss3D()

def train():
    @torch.no_grad()
    def validation(model, loader, loss_fn):
        losses = []
        model.eval()
        bar = tqdm(enumerate(loader), total=len(loader))
        for step, data in bar:
            image, target = data["image"], data["mask"]
            # image = torch.squeeze(image, 0)
            #         target = torch.squeeze(target, 0)
            # print(target.size())
            # optimizer.zero_grad()
            image, target = image.float(), target.float()
            image, target = image.to(device), target.to(device)
            # image = torch.unsqueeze(image, 1)
            # target = torch.unsqueeze(target, 1)
            #         image = torch.unsqueeze(image, 1)
            #         target = torch.unsqueeze(target, 1)
            # image, target = image.to(device), target.float().to(device)
            # target = torch.squeeze(target, 0)
            # target =
            image = torch.unsqueeze(image, 1)
            target = torch.unsqueeze(target, 1)
            # print(target.shape)
            # print(image.shape)
            output = model(image)
            loss = loss_fn(output, target)
            losses.append(loss.item())

        return np.array(losses).mean()

    header = r'''
            Train | Valid
    Epoch |  Loss |  Loss | Time, m
    '''
    #          Epoch         metrics            time
    raw_line = '{:6d}' + '\u2502{:7.3f}' * 2 + '\u2502{:6.2f}'
    print(header)

    EPOCHES = 100
    best_loss = 10
    for epoch in range(1, EPOCHES + 1):
        losses = []
        start_time = time.time()
        model.train()
        for step, data in tqdm(enumerate(train_loader1), total=len(train_loader1)):
            image, target = data["image"], data["mask"]
            # image = torch.squeeze(image, 0)
            #         target = torch.squeeze(target, 0)
            # print(target.size())
            optimizer.zero_grad()
            image, target = image.float(), target.float()
            image, target = image.to(device), target.to(device)
            image = torch.unsqueeze(image, 1)
            target = torch.unsqueeze(target, 1)
            # print(target.shape)
            # print(image.shape)
            output = model(image)
            #         print(output)
            #         print(target)
            # output = model(image)
            # print(output.shape)
            loss = loss_fn(output, target)
            #         loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            # print(loss.item())

        vloss = validation(model, test_loader1, loss_fn)
        print(raw_line.format(epoch, np.array(losses).mean(), vloss,
                              (time.time() - start_time) / 60 ** 1))
        losses = []

        if vloss < best_loss:
            best_loss = vloss
            torch.save(model.state_dict(), 'model_best_unet.pth')
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class EleMic1(Dataset):
  def __init__(self,image, target, transforms=None):
    self.transforms=transforms
    self.target=target
    self.image=image

  def __len__(self):
    return(len(self.image))

  def __getitem__(self, idx):
    image=self.image[idx]
    target = self.target[idx]
    if self.transforms:
        image = self.transforms(image=image, mask=target)["image"]
        image = image.copy()
        mask = self.transforms(image=image, mask=target)["mask"]
    # return (image.to(device), target.to(device))
    return {
        'image': torch.from_numpy(image).to(device).squeeze(),
        'mask': torch.from_numpy(target).to(device).squeeze()
    }
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
from torchmetrics.classification import Dice
dice = Dice()
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
# train_raw = load_volume(path= "../data/train/masks/**.jpg")
# test_raw = load_volume(path= "../data/test/masks/**.jpg")
# train_img, train_mask = train_raw["image"], train_raw["mask"]
# test_img, test_mask = test_raw["image"], test_raw["mask"]
# # for i in range()
# print(np.array(train_raw["image"]).shape)
# train_img, train_mask = random_crop_3d(train_img, train_mask, [64, 128, 128], train=True)
# test_img, test_mask = random_crop_3d(test_img, test_mask, [64, 128, 128], train=False)
# # print(np.array(test_img).shape)
# temp1= list(zip(train_img, train_mask))
# temp2 = list(zip(test_img, test_mask))
# random.shuffle(temp1)
# random.shuffle(temp2)
# res1, res2 = zip(*temp1)
# res3, res4 = zip(*temp2)
# trainSet1, trainSetGround1 = list(res1), list(res2)
# testSet1, testSetGround1 = list(res3), list(res4)
#
# # print(np.array(trainSet).shape)
# train_dataset1 = EleMic1(trainSet1, trainSetGround1)
# test_dataset1 = EleMic1(testSet1, testSetGround1)
# train_loader1 = DataLoader(train_dataset1, batch_size=1, shuffle=True, drop_last= True)
# test_loader1 = DataLoader(test_dataset1, batch_size= 1, shuffle= True, drop_last=True)
# # def loss_fn(y_pred, y_true):
#     bce = bce_fn(y_pred, y_true)
#     dice = dice_fn(y_pred.sigmoid(), y_true)
#     return 0.5 * bce + 0.5 * dice
# from loss_3d import *
# loss_fn = DiceLoss()
if __name__ == '__main__':
    # print(np.array(img2).shape)
    # print(np.array(label2).shape)
    from dataset3d import *
    train_dataset = EMdataset3d(mode= 'train')
    test_dataset = EMdataset3d(mode= 'test')
    train_loader = EMdataset3d()
    # train_dataset = EmDataset(train_img, train_mask, transforms= train_transform)
    # train_loader = DataLoader(train_dataset, batch_size = 2, shuffle = True, drop_last= True)
    # test_dataset = EmDataset(test_img, test_mask, transforms = None)
    # test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=True, drop_last= True)
    # print(f"The train dataset has {len(train_loader)} batch")
    # print(f"The test dataset has {len(test_loader)} batch")
    # #show
    # # dataset = train_dataset
    # # ax = plt.axes(projection='3d')
    # # # ax.plot3D(x, y, z,'gray')
    # # for i in range(len(dataset)):
    # i = np.random.randint(0, len(train_dataset))
    # # # for i in range(0,len(dataset)):
    # data = train_dataset[i]
    # image = data["image"]
    # mask = data["mask"]
    # print(image.shape)
    #
    # image = torch.squeeze(image, 0)
    # mask = torch.squeeze(mask, 0)
    # image = image.cpu().numpy()
    # mask = mask.cpu().numpy()
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
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    #
    # axes[0].imshow(mask[64, :, :], cmap='gray')
    # axes[0].set_title('YZ Plane')
    #
    # axes[1].imshow(mask[:, 64, :], cmap='gray')
    # axes[1].set_title('XZ Plane')
    #
    # axes[2].imshow(mask[:, :, 64], cmap='gray')
    # axes[2].set_title('XY Plane')
    # plt.show()
    train()





