import torch 
from torch import nn 
import torch.nn.functional as F


class Identity(nn.Module):
    """
    Identity block.
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

    def __repr__(self):
        return "{name}()".format(name=self.__class__.__name__)


class BreakBlock(nn.Module):
    """
    Break coonnection block for hourglass.
    """
    def __init__(self):
        super(BreakBlock, self).__init__()

    def forward(self, x):
        return None

    def __repr__(self):
        return "{name}()".format(name=self.__class__.__name__)


class Swish(nn.Module):
    """
    Swish activation function from 'Searching for Activation Functions,' https://arxiv.org/abs/1710.05941.
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class HSigmoid(nn.Module):
    """
    Approximated sigmoid function, so-called hard-version of sigmoid from 'Searching for MobileNetV3,'
    https://arxiv.org/abs/1905.02244.
    """
    def forward(self, x):
        return F.relu6(x + 3.0, inplace=True) / 6.0


class HSwish(nn.Module):
    """
    H-Swish activation function from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters
    ----------
    inplace : bool, default False
        Whether to use inplace version of the module.
    """
    def __init__(self, inplace: bool = False):
        super(HSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0