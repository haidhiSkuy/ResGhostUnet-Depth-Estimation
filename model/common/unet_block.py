import torch 
from torch import nn 
import torch.nn.functional as F
from model.common.ghost_module import GhostConvBlock  

class DownConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__() 
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
    
    def forward(self, x): 
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        pool = self.maxpool(conv2)
        return conv2, pool 

class UpConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=2, padding=0, stride=2) 
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1) 
        self.relu = nn.ReLU() 
    
    def forward(self, x, skip_connection_layer):
        x = self.upconv(x)
        x = torch.cat((skip_connection_layer, x), dim=1) 
        x = F.relu(self.conv1(x)) 
        x = F.relu(self.conv2(x))
        return x 
    
class GhostDownConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__() 
        self.ghost_conv1 = GhostConvBlock(in_channels=in_channel, out_channels=out_channel)
        self.ghost_conv2 = GhostConvBlock(in_channels=out_channel, out_channels=out_channel)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
    
    def forward(self, x): 
        conv1 = self.ghost_conv1(x) 
        conv2 = self.ghost_conv2(conv1)
        pool = self.maxpool(conv2)
        return conv2, pool   
    

class GhostUpConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=2, padding=0, stride=2) 
        self.ghost_conv1 = GhostConvBlock(in_channels=in_channel, out_channels=out_channel) 
        self.ghost_conv2 = GhostConvBlock(in_channels=out_channel, out_channels=out_channel) 
    
    def forward(self, x, skip_connection_layer):
        x = self.upconv(x)
        x = torch.cat((skip_connection_layer, x), dim=1) 
        x = self.ghost_conv1(x)
        x = self.ghost_conv2(x)
        return x
    
class ResGhostDownConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__() 
        self.ghost_conv1 = GhostConvBlock(in_channels=in_channel, out_channels=out_channel)
        self.ghost_conv2 = GhostConvBlock(in_channels=out_channel, out_channels=out_channel)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.shortcut = nn.Sequential(
                GhostConvBlock(in_channels=in_channel, out_channels=out_channel),
                nn.BatchNorm2d(out_channel)
            )
    
    def forward(self, x):
        out = self.ghost_conv1(x)
        out = self.ghost_conv2(out)
        
        out += self.shortcut(x)
    
        pool = self.maxpool(out)
        return out, pool 

class ResGhostUpConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=2, padding=0, stride=2) 
        self.ghost_conv1 = GhostConvBlock(in_channels=in_channel, out_channels=out_channel)
        self.ghost_conv2 = GhostConvBlock(in_channels=out_channel, out_channels=out_channel)

        self.shortcut = nn.Sequential(
                GhostConvBlock(in_channels=in_channel, out_channels=out_channel),
                nn.BatchNorm2d(out_channel)
            )
    
    def forward(self, x, skip_connection_layer):
        up = self.upconv(x)
        up = torch.cat((skip_connection_layer, up), dim=1) 

        out = self.ghost_conv1(up)
        out = self.ghost_conv2(out)

        out += self.shortcut(up)

        return out