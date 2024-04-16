from typing import Optional, List

import numpy as np
import random
import matplotlib.pyplot as plt
import copy
from os import path

import timm
from albumentations.pytorch import ToTensorV2
from monai.metrics.utils import create_table_neighbour_code_to_surface_area
from segmentation_models_pytorch.base import SegmentationHead, Conv2dReLU
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

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

# from models.basic import Conv2dReLU


class SenUNetStem(nn.Module):
    def __init__(self, encoder_name="resnest26d",output_stride=32,
                 encoder_depth=5 , in_chans=1,
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None, classes=1, activation=None):
        super(SenUNetStem, self).__init__()
        kwargs = dict(
            in_chans=in_chans,
            features_only=True,
            # output_stride=output_stride,
            pretrained=True,
            out_indices=tuple(range(encoder_depth)),
        )
        self.conv_stem = Conv2dReLU(in_chans, 16, 3, padding=1)
        self.encoder = timm.create_model(encoder_name, **kwargs)
        self._out_channels = [
            32,
        ] + self.encoder.feature_info.channels()

        self.decoder = UnetDecoder(
            encoder_channels=self._out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1] + 16,
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        # self.n_time = self.n_time
        # self.pickup_index = self.pickup_index

    def forward(self, x):
        B, C, H, W = x.shape
        h = (H//32)*32
        w = (W//32)*32
        x = x[:,:,:h,:w]
        stem = self.conv_stem(x)
        features = self.encoder(x)
        features = [
            stem,
        ] + features

        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        masks = F.pad(masks,[0,W-w,0,H-h,0,0,0,0], mode='constant', value=0)

        return masks[:,0]


# if __name__ == '__main__':
#     # model = SenUNetStem()
#     # print('model type: ', model.__class__.__name__)
#     # num_params = sum([p.data.nelement() for p in model.parameters()])
#     # print('number of trainable parameters: ', num_params)
#     # x = torch.randn(1, 18, 64, 64)
#     # y = model(x)
#     # print(x.size(), y.size())