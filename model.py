
import torch
import torch.nn as nn

import math

import torch
import torch.nn as nn
import torch.nn.functional as F



class conv_3d(nn.Module):
    def __init__(self):
        super(conv_3d, self).__init__()
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
        return self._features(x)

class conv_3d(nn.Module):
    def __init__(self):
        super(conv_3d, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=5, padding=1),
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
        out = self._features(x)
        print("conv3d", out.shape)
        out= out.reshape(out.shape[0], out.shape[1]*out.shape[2], out.shape[3], out.shape[4])
        print("reshape conv3d ",out.shape)
        return out

class conv_2d(nn.Module):
    def __init__(self):
        super(conv_2d, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(640, 640, kernel_size = 3, padding =1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
            
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(640, 512, kernel_size = 3, padding =1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size = 3, padding =1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        
    
    def forward(self,x):
        out = self.conv2(self.conv1(x))
        out = self.conv3(out)
        print(out.shape)
        out = out.view(out.shape[0],-1)
        print(out.shape)
        return out







