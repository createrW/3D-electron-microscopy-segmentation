import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm


def calculateIoU(gtMask, predMask):
    # Calculate the true positives,
    # false positives, and false negatives
    tp = 0
    fp = 0
    fn = 0
    gtMask = gtMask.squeeze().squeeze()
    predMask = predMask.squeeze().squeeze()
    for i in range(len(gtMask)):
        for j in range(len(gtMask[0])):
            for k in range((len(gtMask[0][0]))):
                if gtMask[i][j][k] == 1 and predMask[i][j][k] == 1:
                    tp += 1
                elif gtMask[i][j][k] == 0 and predMask[i][j][k] == 1:
                    fp += 1
                elif gtMask[i][j][k] == 1 and predMask[i][j][k] == 0:
                    fn += 1

    # Calculate IoU
    iou = tp / (tp + fp + fn)

    return iou
from The_final_unet3d import train_dataloader, test_dataloader
from my_3D.unet3d_2 import *
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = UNet(1, 1, 1)
model = model.to(device)
state_dict = torch.load("./model_best_3d.pth_BounaryDOU")

model.load_state_dict(state_dict)

bar = tqdm(enumerate(train_dataloader), total = len(train_dataloader))
nums = 0
model.to(device)
model.eval()
raw_line = '\u2502{:7.3f}' + '\u2502{:6.2f}' * 2 + '\u2502{:7.3f}'
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

def method4(inputs, target):

    intersection = (inputs * target).sum()
    union = inputs.sum() + target.sum()
    dice = (2. * intersection) / (union + 1e-8)


    return dice
bce_fn = nn.BCEWithLogitsLoss()
dice_fn = SoftDiceLoss()
from torchmetrics.classification import Dice, BinaryJaccardIndex

dice = Dice()

def loss_fn(y_pred, y_true):
    bce = bce_fn(y_pred, y_true)
    dice = dice_fn(y_pred.sigmoid(), y_true)
    return 0.5 * bce + 0.5 * dice

metric = BinaryJaccardIndex()
losses = []
dices = []
ious = []
new_ious = []
num_correct = 0.0
num_pixels = 0.0
dice_score = 0.0
new_iou = 0.0
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
        # print(f"output is {output.shape}")
        # print(f"mask is  {mask.shape}")
        iou = metric(((nn.Sigmoid()(output).cpu())), mask.cpu())
        output = torch.sigmoid(output)
        output = (output > 0.5).cpu().float()
        # output = nn.Sigmoid()(output).cpu()
        mask = mask.cpu()
        new_iou = calculateIoU(output, mask)
        new_ious.append(new_iou)
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
                              np.array(dices).mean(), np.array(ious).mean()), np.array(new_ious).mean())
    print(np.array(new_ious).mean())
