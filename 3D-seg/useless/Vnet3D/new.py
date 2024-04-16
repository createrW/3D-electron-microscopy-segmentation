import math
import torch
# from config import (
#     TRAINING_EPOCH, NUM_CLASSES, IN_CHANNELS, BCE_WEIGHTS, BACKGROUND_AS_CLASS, TRAIN_CUDA
# )
from torch.nn import CrossEntropyLoss
# from dataset import get_train_val_test_Dataloaders
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
# from unet3d import UNet3D
# from transforms import (train_transform, train_transform_cuda,
#                         val_transform, val_transform_cuda)
from my_3D.unet3d_2 import *
model = UNet(1, 1,1 )

# if BACKGROUND_AS_CLASS: NUM_CLASSES += 1

writer = SummaryWriter("runs")

# model = UNet3D(in_channels=1, num_classes=1)
# train_transforms = train_transform
# val_transforms = val_transform

if torch.cuda.is_available():
    model = model.cuda()
    # train_transforms = train_transform_cuda
    # val_transforms = val_transform_cuda
elif not torch.cuda.is_available():
    print('cuda not available! Training initialized on cpu ...')
#
# train_dataloader, val_dataloader, _ = get_train_val_test_Dataloaders(train_transforms=train_transforms,
#                                                                      val_transforms=val_transforms,
#                                                                      test_transforms=val_transforms)
from preprocess.final import *

# BCE_WEIGHTS = [0.004, 0.996]
criterion = CrossEntropyLoss()
optimizer = Adam(params=model.parameters())

min_valid_loss = math.inf

for epoch in range(20):

    train_loss = 0.0
    model.train()
    for data in train_loader1:
        image, ground_truth = data['image'], data['mask']
        optimizer.zero_grad()
        image = torch.unsqueeze(image, 1)
        ground_truth = torch.unsqueeze(ground_truth, 1)
        target = model(image)
        loss = criterion(target, ground_truth)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    valid_loss = 0.0
    model.eval()
    for data in test_loader1:
        image, ground_truth = data['image'], data['mask']
        image = torch.unsqueeze(image, 1)
        ground_truth = torch.unsqueeze(ground_truth, 1)
        target = model(image)
        loss = criterion(target, ground_truth)
        valid_loss = loss.item()

    writer.add_scalar("Loss/Train", train_loss / len(train_loader1), epoch)
    writer.add_scalar("Loss/Validation", valid_loss / len(test_loader1), epoch)

    print(
        f'Epoch {epoch + 1} \t\t Training Loss: {train_loss / len(train_loader1)} \t\t Validation Loss: {valid_loss / len(test_loader1)}')

    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        torch.save(model.state_dict(), f'checkpoints/epoch{epoch}_valLoss{min_valid_loss}.pth')

writer.flush()
writer.close()
