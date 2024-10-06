"""
    GhostNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'GhostNet: More Features from Cheap Operations,' https://arxiv.org/abs/1911.11907.
"""

import math
import torch
import torch.nn as nn
from model.common.common import (conv1x1_block, dwconv3x3_block)


class GhostHSigmoid(nn.Module):
    """
    Approximated sigmoid function, specific for GhostNet.
    """

    def forward(self, x):
        return torch.clamp(x, min=0.0, max=1.0)


class GhostConvBlock(nn.Module):
    """
    GhostNet specific convolution block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            activation=(lambda: nn.ReLU(inplace=True))
        ):
        super(GhostConvBlock, self).__init__()
        main_out_channels = math.ceil(0.5 * out_channels)
        cheap_out_channels = out_channels - main_out_channels

        self.main_conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=main_out_channels,
            activation=activation
        )
        
        self.cheap_conv = dwconv3x3_block(
            in_channels=main_out_channels,
            out_channels=cheap_out_channels,
            activation=activation
        )

    def forward(self, x):
        x = self.main_conv(x)
        y = self.cheap_conv(x)
        return torch.cat((x, y), dim=1)

if __name__ == "__main__": 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device}")

    ghost_module = GhostConvBlock(
        in_channels=3, 
        out_channels=64
    ) 

    sample = torch.randn(1, 3, 224, 224) 
    out = ghost_module(sample)

    print(out.size())