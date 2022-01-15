import torch
import torch.nn as nn
import torch.nn.functional as F

class Double_conv_3d(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv_3d = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv3d(out_channels, out_channels, 3, padding = 1),
            nn.ReLU(inplace = True)
            )

    def forward(self, x):
        return self.double_conv_3d(x)

class Unet_3d(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Unet_3d, self).__init__()

        self.down1 = Double_conv_3d(in_ch, 32)
        self.down2 = Double_conv_3d(32, 64)
        self.down3 = Double_conv_3d(64, 128)

        self.pool = nn.MaxPool3d(2)
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size = 2, stride = 2),
            nn.Conv3d(64, 64, kernel_size = (2,1,1), padding = (1,0,0)),
            nn.ReLU(inplace = True)
            )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size = 2, stride = 2),
            nn.Conv3d(32, 32, kernel_size = (2,1,1), padding = (1,0,0)),
            nn.ReLU(inplace = True)
            )

        self.up1 = Double_conv_3d(128, 64)
        self.up2 = Double_conv_3d(64, 32)

        self.conv_last = nn.Conv3d(32, out_ch, kernel_size = 1)
        self.afn_last = nn.Tanh()


    def forward(self, x):
        inputs = x 

        conv1 = self.down1(x)
        x = self.pool(conv1)
        conv2 = self.down2(x)
        x = self.pool(conv2)
        conv3 = self.down3(x)

        x = self.upsample1(conv3)
        x = torch.cat([x, conv2], dim=1)

        x = self.up1(x)

        x = self.upsample2(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.up2(x)

        x = self.conv_last(x)
        x = self.afn_last(x)

        out = x + inputs

        return out

    
