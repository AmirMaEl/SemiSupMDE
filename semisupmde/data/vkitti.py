"""Virtual KITTI 1.3.1 source data for finetuning."""
import os
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from PIL import Image

from .kitti import read_split, to_tensor

MAX_DEPTH_CM = 8000  # vKITTI depth is stored in cm; clip at 80 m


class VKittiPairs(data.Dataset):
    """
    Source pairs for the pairwise loss (Eq. 4-7): the indexed vKITTI image and an independently drawn one,
    each with its depth (clipped to 80 m and scaled to [-1, 1]).
    """
    def __init__(self, split_file, root, img_size=(640, 192), flip=True, rotation=True, color_jitter=0.4):
        self.paths = read_split(split_file, os.path.join(root, 'vkitti_1.3.1_rgb'))
        self.img_size = img_size
        self.flip, self.rotation, self.color_jitter = flip, rotation, color_jitter

    def load(self, path):
        img = Image.open(path).convert('RGB')
        depth = Image.open(path.replace('vkitti_1.3.1_rgb', 'vkitti_1.3.1_depthgt'))
        if self.flip and random.random() > 0.5:
            img, depth = F.hflip(img), F.hflip(depth)
        if self.rotation and random.random() > 0.5:
            degree = random.randrange(-500, 500) / 100
            img = F.rotate(img, degree, interpolation=InterpolationMode.BICUBIC)
            depth = F.rotate(depth, degree, interpolation=InterpolationMode.NEAREST)
        img = img.resize(self.img_size, Image.BICUBIC)
        depth = depth.resize(self.img_size, Image.NEAREST)
        if self.color_jitter > 0:
            c = self.color_jitter
            img = transforms.ColorJitter(brightness=c, contrast=c, saturation=c, hue=min(c, 0.5))(img)
        depth = torch.clamp(transforms.ToTensor()(depth), 0, MAX_DEPTH_CM) / MAX_DEPTH_CM * 2 - 1
        return to_tensor(img), depth

    def __getitem__(self, index):
        x1, y1 = self.load(self.paths[index])
        x2, y2 = self.load(self.paths[random.randrange(len(self.paths))])
        return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}

    def __len__(self):
        return len(self.paths)
