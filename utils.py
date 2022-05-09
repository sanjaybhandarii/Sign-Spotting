import torch
from torchvision.transforms import Compose, Lambda, Grayscale
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
)
from torch.utils.data import DataLoader
import numpy as np
from dataset import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MAGE_HEIGHT = 512
IMAGE_WIDTH = 512
IMAGE_CHANNEL = 3
NUM_FRAMES = 25
NUM_CLASSES = 60
mean = 0
std = 0



inputs =[] #x
classes = [] #y

transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            
            Lambda(lambda x: x.permute(1,0,2,3)),#(frames(depth), channel, height, width) -> (channel, frames(depth), height, width)
            
            UniformTemporalSubsample(NUM_FRAMES),
            Lambda(lambda x: x.permute(1,0,2,3)),#(frames(depth), channel, height, width)
            
            Lambda(lambda x: x/255.0), 
            NormalizeVideo((mean,), (std,)),
            CenterCropVideo([720,800]),
            Lambda(lambda x: x.permute(1,0,2,3)),#(channel, frames(depth), height, width)
            
            
        ]
        
    ),
)



    # Initialize an EncodedVideo helper class
video = EncodedVideo.from_path("/home/chaos/Documents/GitHub/Sign-Spotting/p01_n000.mp4")
    
def get_data_info(f):
    for line in f:
        a = line.split(',')
        yield a
        


with open('/home/chaos/Documents/GitHub/Sign-Spotting/p01_n000.txt') as f: 
    for x in get_data_info(f):
        classes.append(x[0])
        start_time = x[1]
        end_time = x[2]
        
        video_data = video.get_clip(start_sec=float(start_time)/1000.0, end_sec=float(end_time)/1000.0)

        
        video_data["video"] = Grayscale(num_output_channels=1)((video_data["video"]).permute(1,0,2,3))

        #print(video_data["video"].shape)
        std, mean = torch.std_mean(video_data["video"])
        #print(std, mean)
        video_data = transform( video_data)

    # Move the inputs to the desired device
        inputs.append(video_data["video"])


ds = SignDataset(inputs, classes)
print(len(ds))


dataloader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=1)
x,y =((iter(dataloader).next()))
print(x.shape)

