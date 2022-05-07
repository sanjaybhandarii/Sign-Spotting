import torch
import pytorchvideo
import torch.utils.data
from torchvision.transforms import Compose, Lambda, Grayscale
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
    UniformCropVideo
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MAGE_HEIGHT = 512
IMAGE_WIDTH = 512
IMAGE_CHANNEL = 3
NUM_FRAMES = 25
NUM_CLASSES = 60
alpha = 4


def transform_video(std, mean, video):
    transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(NUM_FRAMES),
            Lambda(lambda x: x/255.0),
            NormalizeVideo([mean for _ in range(3)] , [std for _ in range(3)]),
            
            CenterCropVideo(512),
            
        ]
        
    ),
    )
    return (transform(video))

inputs =[]




    # Initialize an EncodedVideo helper class
video = EncodedVideo.from_path("/home/chaos/Documents/GitHub/Sign-Spotting/p01_n000.mp4")
    
def get_data_info(f):
    for line in f:
        a = line.split(',')
        yield a
        

thisdict = {}
with open('/home/chaos/Documents/GitHub/Sign-Spotting/p01_n000.txt') as f: 
    for x in get_data_info(f):
        cls = x[0]
        start_time = x[1]
        end_time = x[2]
        
        # video_data = video.get_clip(start_sec=float(start_time)/1000.0, end_sec=float(end_time)/1000.0)
        video_data = video.get_clip(start_sec=float(start_time)/1000.0, end_sec=float(end_time)/1000.0)

        video_data["video"] = video_data["video"].unsqueeze(0).permute(0,2,1,3,4)

        print(video_data["video"].shape)
        video_data["video"] = Grayscale(num_output_channels=1)(video_data["video"].unsqueeze(0))
        std, mean = torch.std_mean(video_data["video"])
        print(std, mean)
        video_data = transform_video(std, mean, video_data)

    # Move the inputs to the desired device
        # inputs.append(video_data["video"])


# inputs = [i.to(device)[None, ...] for i in inputs]
    