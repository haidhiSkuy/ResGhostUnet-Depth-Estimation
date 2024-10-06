import torch 
from torch import nn 
from model.common.unet_block import ResGhostDownConv, ResGhostUpConv


class ResidualGhostUNet(nn.Module):
    def __init__(self, in_channel=3):
        super().__init__()
        # encoder
        self.encoder1 = ResGhostDownConv(in_channel, 64)
        self.encoder2 = ResGhostDownConv(64, 128)
        self.encoder3 = ResGhostDownConv(128, 256)
        self.encoder4 = ResGhostDownConv(256, 512)
        
        # bottleneck 
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # decoder 
        self.decoder1 = ResGhostUpConv(1024, 512)
        self.decoder2 = ResGhostUpConv(512, 256)
        self.decoder3 = ResGhostUpConv(256, 128)
        self.decoder4 = ResGhostUpConv(128, 64)

        # output 
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, padding=0)

        
    def forward(self, x): 
        #ENCODER
        sc1, en1 = self.encoder1(x) #layer for skip connection, maxpool output
        sc2, en2 = self.encoder2(en1)
        sc3, en3 = self.encoder3(en2)
        sc4, en4 = self.encoder4(en3)
   
        #BOTTLENECK
        bottleneck = self.bottleneck(en4)
        
        #DECODER
        upconv1 = self.decoder1(bottleneck,sc4)
        upconv2 = self.decoder2(upconv1,sc3) 
        upconv3 = self.decoder3(upconv2, sc2)
        upconv4 = self.decoder4(upconv3, sc1)

        #OUTPUT 
        output = self.output(upconv4)
        return output