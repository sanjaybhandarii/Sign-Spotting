import torch
import pytorchvideo
import torch.utils.data
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
    UniformCropVideo
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_CHANNEL = 3
NUM_FRAMES = 25
NUM_CLASSES = 60

class SignDataset()

