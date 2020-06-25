import imageio
from scipy import ndimage
from scipy.misc import imresize
import torchvision.transforms.functional as functional
import torchvision.transforms as transforms
import torch
import os
import numpy as np

def load_images(file_path):
    """Loads the specified image set.

    Args:
        file_path (str): The path to the file to load.
            Should be a format that ffmpeg can handle.

    Returns:
        List[FloatTensor]: the frames of the video as a list of 3D tensors
            (channels, width, height)"""

    frames = []
    file_names = os.listdir(file_path)
    file_names = [(int(os.path.splitext(name)[0]), name) for name in file_names]
    file_names = sorted(file_names)
    frames = [ndimage.imread(os.path.join(file_path, name[1])) for name in file_names]
    return frames

    
def load_video(filename):
    """Loads the specified video using ffmpeg.

    Args:
        filename (str): The path to the file to load.
            Should be a format that ffmpeg can handle.

    Returns:
        List[FloatTensor]: the frames of the video as a list of 3D tensors
            (channels, width, height)"""

    vid = imageio.get_reader(filename,  'ffmpeg')
    frames = []
    for i in range(29):
        image = vid.get_data(i)
        frames.append(image)
    return frames
    
def bbc(vidframes, augmentation=False):
    """Preprocesses the specified list of frames by center cropping.
    This will only work correctly on videos that are already centered on the
    mouth region, such as LRITW.

    Args:
        vidframes (List[FloatTensor]):  The frames of the video as a list of
            3D tensors (channels, width, height)

    Returns:
        FloatTensor: The video as a temporal volume, represented as a 5D tensor
            (batch, channel, time, width, height)"""

    temporalvolume = torch.FloatTensor(29,3,256,256)

    croptransform = transforms.CenterCrop((112, 112))

    for i in range(0, 29):
        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])(vidframes[i])

        temporalvolume[i] = result

    return temporalvolume
