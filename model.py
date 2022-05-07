import imp
import torch
import torch.nn as nn


class 3D_conv(nn.Module):
    def __init__(self):
        super(3D_conv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        

        self._features = nn.Sequential(
            self.conv1,
            self.conv2
        )


    def forward(self, x):
        x = x.view((1, 3, 256, 256, 29)) # batch_size, channel,depth, height, width(fps) 
        return self._features(x)