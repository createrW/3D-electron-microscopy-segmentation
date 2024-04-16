import torch
from torch import nn


class DenseBlock(nn.Module):
    """
    Repeatable Dense block as specified by the paper
    This is composed of a pointwise convolution followed by a depthwise separable convolution
    After each convolution is a BatchNorm followed by a ReLU

    Some notes on architecture based on the paper:
      - The first block uses an input channel of 96, and the remaining input channels are 32
      - The hidden channels is always 128
      - The output channels is always 32
      - The depth is always 3
    """

    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int, count: int
    ):
        """
        Create the layers for the dense block

        :param in_channels:      number of input features to the block
        :param hidden_channels:  number of output features from the first convolutional layer
        :param out_channels:     number of output features from this entire block
        :param count:            number of times to repeat
        """
        super().__init__()

        # First iteration takes different number of input channels and does not repeat
        first_block = [
            nn.Conv3d(in_channels, hidden_channels, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(),
            nn.Conv3d(
                hidden_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        ]

        # Remaining repeats are identical blocks
        repeating_block = [
            nn.Conv3d(out_channels, hidden_channels, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(),
            nn.Conv3d(
                hidden_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        ]

        self.convs = nn.Sequential(
            *first_block,
            *repeating_block * (count - 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the block

        :param x:  image tensor
        :return:   output of the forward pass
        """
        return self.convs(x)

class TransitionBlock(nn.Module):
    """
    Transition Block (transition layer) as specified by the paper
    This is composed of a pointwise convolution followed by a pointwise convolution with higher stride to reduce the image size
    We use BatchNorm and ReLU after the first convolution, but not the second

    Some notes on architecture based on the paper:
      - The number of channels is always 32
      - The depth is always 3
    """

    def __init__(self, channels: int):
        """
        Create the layers for the transition block

        :param channels:  number of input and output channels, which should be equal
        """
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(channels),
            nn.ReLU(),
            # This conv layer is analogous to H-Dense-UNet's 1x2x2 average pool
            nn.Conv3d(channels, channels, kernel_size=(1, 1, 1), stride=(1, 2, 2)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the block

        :param x:  image tensor
        :return:   output of the forward pass
        """
        return self.convs(x)


from typing import Tuple

import torch
from torch import nn


class UpsamplingBlock(nn.Module):
    """
    Upsampling Block (upsampling layer) as specified by the paper
    This is composed of a 2d bilinear upsampling layer followed by a convolutional layer, BatchNorm layer, and ReLU activation
    """

    def __init__(self, in_channels: int, out_channels: int, size: Tuple):
        """
        Create the layers for the upsampling block

        :param in_channels:   number of input features to the block
        :param out_channels:  number of output features from this entire block
        :param scale_factor:  tuple to determine how to scale the dimensions
        :param residual:      residual from the opposite dense block to add before upsampling
        """
        super().__init__()
        # blinear vs trilinear kernel size and padding
        if size[0] == 2:
            d_kernel_size = 3
            d_padding = 1
        else:
            d_kernel_size = 1
            d_padding = 0

        self.upsample = nn.Upsample(
            scale_factor=size, mode="trilinear", align_corners=True
        )
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(d_kernel_size, 3, 3),
                padding=(d_padding, 1, 1),
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, projected_residual):
        """
        Forward pass through the block

        :param x:  image tensor
        :return:   output of the forward pass
        """
        residual = torch.cat(
            (self.upsample(x), self.upsample(projected_residual)),
            dim=1,
        )
        return self.conv(residual)

