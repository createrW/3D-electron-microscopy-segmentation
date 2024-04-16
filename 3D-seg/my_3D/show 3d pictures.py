import torch

from The_final_unet3d import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
dataloader = train_dataloader
bar = tqdm(enumerate(test_dataloader), total = len(test_dataloader))
num = 0
from my_3D.unet3d_2 import *
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = UNet(1, 1, 1)
model = model.to(device)
state_dict = torch.load("./model_best_3d.pth_BounaryDOU")

model.load_state_dict(state_dict)
with torch.no_grad():
    for idx, data in bar:
        if(num==5):
            break
        image, mask = data["image"].to(device), data["mask"].float().to(device)
        image = torch.unsqueeze(image, 1)
        output = model(image)
        image = image.cpu().float()
        mask = mask.cpu().float()
        image = torch.squeeze(image, 0)
        output = torch.squeeze(output, 0)
        image = torch.squeeze(image, 0)
        output = torch.squeeze(output, 0)
        mask = torch.squeeze(mask, 0)
        print(output.shape)
        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        output = torch.sigmoid(output)
        output = (output > 0.5).cpu().float()
        output = np.transpose(output, (2, 0, 1))
        fig, axes = plt.subplots(1, 3, figsize=(11,3))
        # print(image[64, :, :].shape)
        fig.suptitle("YZ plane")
        axes[0].imshow(image[64, :, :], cmap='bone')
        axes[0].set_title('Model Input')

        axes[1].imshow(mask[64, :, :], cmap='bone')
        axes[1].set_title('Real Output')

        axes[2].imshow(output[64, :, :], cmap='bone')
        axes[2].set_title('Model Output')

        plt.tight_layout()
        plt.show()
        fig, axes = plt.subplots(1, 3, figsize=(11,3))
        # print(image[64, :, :].shape)
        fig.suptitle("XZ plane")
        axes[0].imshow(image[:, 64, :], cmap='bone')
        axes[0].set_title('Model Input')

        axes[1].imshow(mask[:, 64, :], cmap='bone')
        axes[1].set_title('Real Output')

        axes[2].imshow(output[:, 64, :], cmap='bone')
        axes[2].set_title('Model Output')

        plt.tight_layout()
        plt.show()
        fig, axes = plt.subplots(1, 3, figsize=(11,3))
        # print(image[64, :, :].shape)
        fig.suptitle("XY plane")
        axes[0].imshow(image[:, :, 64], cmap='bone')
        axes[0].set_title('Model Input')

        axes[1].imshow(mask[:, :, 64], cmap='bone')
        axes[1].set_title('Real Output')

        axes[2].imshow(output[:, :, 64], cmap='bone')
        axes[2].set_title('Model Output')

        plt.tight_layout()
        plt.show()
        num+=1