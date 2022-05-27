import torch
from torchvision.transforms import Compose, Lambda, Grayscale, Normalize
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
)
import pickle
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



IMAGE_HEIGHT = 720
IMAGE_WIDTH = 800
IMAGE_CHANNEL = 1
NUM_FRAMES = 41
NUM_CLASSES = 60



inputs =[] #x
classes = [] #y

def pad_constant(tensor, length, value):
    return torch.cat([tensor, tensor.new_zeros(length - tensor.size(0), *tensor.size()[1:])], dim=0)


def transform_data(x, mean, std):
    
    transform =  ApplyTransformToKey(
        key="video",
        transform=Compose(
            [

                # Lambda(lambda x: x.permute(1,0,2,3)),#(frames(depth), channel, height, width) -> (channel, frames(depth), height, width)

                # UniformTemporalSubsample(NUM_FRAMES),
                # Lambda(lambda x: x.permute(1,0,2,3)),#(frames(depth), channel, height, width)
                Lambda(lambda x: pad_constant(x, NUM_FRAMES, 0)),
                Lambda(lambda x: x/255.0),
                
                Normalize((mean,), (std,)),

                CenterCropVideo([720,800]),
                Lambda(lambda x: x.permute(1,0,2,3)),#(channel, frames(depth), height, width)

            ]

        ),
    )
    
    return transform(x)



    
        


def load_dataloader(batch_size):
    with open('path/MSSL_TRAIN_SET_GT.pkl', 'rb') as f:
        data = pickle.load(f)



    for key in data.keys():
        filename = key
        print("file",filename)

    # file functions

        for x in data[key]:
            classes.append(x[0])
            start_time = x[1]
            end_time = x[2]
   #give path
        video = EncodedVideo.from_path("path/"+filename)
    
            
            
        video_data = video.get_clip(start_sec=float(start_time)/1000.0, end_sec=float(end_time)/1000.0)

            
        video_data["video"] = Grayscale(num_output_channels=1)((video_data["video"]).permute(1,0,2,3))
#             video_data["video"] = video_data["video"]/255
            #print(video_data["video"].shape)
            
        std, mean = torch.std_mean(video_data["video"])
        std = std/255.0
        mean = mean/255.0
        print(std, mean)
            
            
        video_data = transform_data( video_data, mean, std)

        # Move the inputs to the desired device
        inputs.append(video_data["video"])

    signds = SignDataset(inputs, classes)
    dataloader = DataLoader(signds, batch_size=batch_size, shuffle=True, num_workers=1)

    return dataloader
        

