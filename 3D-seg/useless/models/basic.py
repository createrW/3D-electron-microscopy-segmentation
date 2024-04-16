import sys, os
import cv2
import pandas as pd
from glob import glob
import numpy as np

from timeit import default_timer as timer
# !python -m pip install --no-index --find-links=/kaggle/input/pip-download-for-segmentation-models-pytorch segmentation-models-pytorch
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import LayerNorm2d
from torch.cuda.amp import autocast
import matplotlib
import matplotlib.pyplot as plt




from typing import Optional, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder


class SenUNet(nn.Module):
    def __init__(self, encoder_name="resnest26d", output_stride=32,
                 encoder_depth=5, n_time=8, pickup_index=4,
                 decoder_use_batchnorm: bool = True,
                 decoder_channels: List[int] = (256, 128, 64, 32, 16),
                 decoder_attention_type: Optional[str] = None, classes=1, activation=None):
        super(SenUNet, self).__init__()
        kwargs = dict(
            in_chans=1,
            features_only=True,
            #             output_stride=output_stride,
            pretrained=False,
            out_indices=tuple(range(encoder_depth)),
        )
        self.encoder = timm.create_model(encoder_name, **kwargs)
        self._out_channels = [
                                 3,
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
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        # not all models support output stride argument, drop it by default
        #         if output_stride == 32:
        #             kwargs.pop("output_stride")

        # self.conv4_3d_1 = Residual3DBlock(512)
        self.n_time = n_time
        self.pickup_index = pickup_index

    def forward(self, x):
        B, C, H, W = x.shape
        h = (H // 32) * 32
        w = (W // 32) * 32
        x = x[:, :, :h, :w]
        features = self.encoder(x)
        features = [
                       x,
                   ] + features

        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)
        masks = F.pad(masks, [0, W - w, 0, H - h, 0, 0, 0, 0], mode='constant', value=0)

        return masks


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_layernorm=True,
    ):



        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_layernorm),
        )
        relu = nn.GELU()



        if use_layernorm and use_layernorm != "inplace":
            bn = LayerNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class SenUNetStem(nn.Module):
    def __init__(self, encoder_name="resnest26d", output_stride=32,
                 encoder_depth=5, n_time=8, pickup_index=4,
                 decoder_use_batchnorm: bool = True,
                 decoder_channels: List[int] = (256, 128, 64, 32, 16),
                 decoder_attention_type: Optional[str] = None, classes=1, activation=None):
        super(SenUNetStem, self).__init__()
        kwargs = dict(
            in_chans=1,
            features_only=True,
            # output_stride=output_stride,
            pretrained=False,
            out_indices=tuple(range(encoder_depth)),
        )
        self.conv_stem = Conv2dReLU(1, 16, 3, use_layernorm=False)
        self.encoder = timm.create_model(encoder_name, pretrained=True,**kwargs)
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
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        # not all models support output stride argument, drop it by default
        # if output_stride == 32:
        #     kwargs.pop("output_stride")

        # self.conv4_3d_1 = Residual3DBlock(512)
        self.n_time = n_time
        self.pickup_index = pickup_index

    def forward(self, x):
        B, C, H, W = x.shape
        h = (H // 32) * 32
        w = (W // 32) * 32
        x = x[:, :, :h, :w]
        stem = self.conv_stem(x)
        features = self.encoder(x)
        features = [
                       stem,
                   ] + features

        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)
        masks = F.pad(masks, [0, W - w, 0, H - h, 0, 0, 0, 0], mode='constant', value=0)

        return masks


def remove_small_objects(mask, min_size):
    # Find all connected components (labels)
    num_label, label, stats, centroid = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # create a mask where small objects are removed
    processed = np.zeros_like(mask)
    for l in range(1, num_label):
        if stats[l, cv2.CC_STAT_AREA] >= min_size:
            processed[label == l] = 255

    return processed

def rle_encode(mask):
    pixel = mask.flatten()
    pixel = np.concatenate([[0], pixel, [0]])
    run = np.where(pixel[1:] != pixel[:-1])[0] + 1
    run[1::2] -= run[::2]
    rle = ' '.join(str(r) for r in run)
    if rle == '':
        rle = '1 0'
    return rle

#-------------------------------

checkpoint_file = ""

net = SenUNetStem(encoder_name="maxvit_tiny_tf_512.in1k")
#run_check_net()
# state_dict = torch.load("/kaggle/input/sennet-models/baseline58_ema.pth", map_location=lambda storage, loc: storage)
# print(net.load_state_dict(state_dict, strict=False))  # True

net = net.eval()
net = net.cuda()
#net = torch.compile(net)

predict_mode = "tile"