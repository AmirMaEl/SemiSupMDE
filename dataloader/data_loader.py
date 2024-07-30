import random
import os
from PIL import Image
import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from params import BS, IMG_SIZE, KITTI_ROOT
from util.kitti_util import KITTI
from dataloader.image_folder import make_dataset

class CreateDataset(data.Dataset):
    """
    Dataset class for loading and transforming KITTI dataset images and depth maps.
    """
    def initialize(self, img_target_file, isTrain=False, mode='unpaired', loadSize=IMG_SIZE):
        self.mode = mode
        self.loadSize = loadSize
        self.img_target_paths, self.img_target_size = make_dataset(img_target_file)
        self.transform_augment = get_transform(isTrain, True)
        self.transform_no_augment = get_transform(isTrain, False)
        self.transform_depth_no_augment = get_depth_transform(isTrain, False)
        print('Loaded KITTI dataset with size:', len(self.img_target_paths))
        self.kitti = KITTI()
        self.isTrain = isTrain

    def __getitem__(self, item):
        if self.mode == 'paired':
            img_target_path = self.img_target_paths[item % self.img_target_size]
        elif self.mode == 'unpaired':
            img_target_path = self.img_target_paths[item]
            # print(f"Loading image: {img_target_path}")
            img = cv2.imread(img_target_path)
            if img is None:
                print(f"Failed to load image: {img_target_path}")
        else:
            raise ValueError('Data mode [%s] is not recognized' % self.opt.dataset_mode)

        img_target = Image.open(img_target_path).convert('RGB')
        w, h = img_target.size if not self.isTrain else (640, 192)
        
        if 'image_02' in img_target_path:
            velo_path = img_target_path.replace("image_02", "velodyne_points")
        elif 'image_03' in img_target_path:
            velo_path = img_target_path.replace("image_03", "velodyne_points")
        else:
            raise ValueError('Data mode [%s] is not recognized' % self.opt.dataset_mode)

        velo_path = velo_path.replace("png", "bin")
        calib_path = "/".join(img_target_path.split('/')[:-4])
        gt, gt_interp = self.kitti.get_depth(calib_path, velo_path, [h, w], interp=True)

        gt_pil = Image.fromarray(gt)
        gt_interp_pil = Image.fromarray(gt_interp)
        img_target = img_target.resize(self.loadSize, Image.BICUBIC)
        label_r_interp = gt_interp_pil.resize(self.loadSize, Image.BILINEAR)
        label_r = gt_pil.resize(self.loadSize, Image.NEAREST, reducing_gap=2.0)
        img_target = self.transform_augment(img_target)

        gt_pil = self.transform_depth_no_augment(label_r) / 50
        gt_pil = (gt_pil * 2) - 1
        gt_interp_pil = self.transform_depth_no_augment(label_r_interp) / 50
        gt_interp_pil = (gt_interp_pil * 2) - 1
        
        filename = '/'.join(img_target_path.split('/')[-5:]).replace('/', '~')

        return {
            'filename': filename,
            'img_target': img_target,
            'lab_target': gt_interp_pil,
            'gt': gt,
            'img_target_paths': img_target_path,
            'image': img_target,
            'depth_interp': gt_interp_pil,
            'depth': gt_pil,
        }

    def __len__(self):
        return self.img_target_size

class KittiModule():
    """
    Module for managing DataLoader for KITTI dataset.
    """
    def __init__(self, batch_size=1, shuffle=True, num_workers=8, drop_last=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
    
    def train_dataloader(self):
        dataset = CreateDataset()
        dataset.initialize('./datasplit/train_kitti.txt', True, loadSize=IMG_SIZE)
        return data.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, drop_last=self.drop_last)
    
    def val_dataloader(self):
        dataset = CreateDataset()
        dataset.initialize('./datasplit/eigen_val.txt', False, loadSize=IMG_SIZE)
        return data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=self.drop_last)
    
    def test_dataloader(self):
        dataset = CreateDataset()
        dataset.initialize('./datasplit/test_kitti.txt', False, loadSize=IMG_SIZE)
        return data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, drop_last=self.drop_last)
    

def get_transform(isTrain, augment):
    """
    Get image transformation operations.
    """
    transforms_list = []
    if augment and isTrain:
        transforms_list.append(transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0))
    transforms_list += [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transforms_list)

def get_depth_transform(isTrain, augment):
    """
    Get depth map transformation operations.
    """
    transforms_list = []
    if augment and isTrain:
        transforms_list.append(transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0))
    transforms_list += [transforms.ToTensor()]
    return transforms.Compose(transforms_list)

def list_img_files_recursively(path, contains=None, path_contains=None):
    """
    Recursively list image files in a directory.
    """
    results = []
    contains = contains or ''
    path_contains = path_contains or ''
    for e in os.listdir(path):
        full_path = os.path.join(path, e)
        ext = e.split('.')[-1]
        if '.' in e and ext.lower() in ['jpg', 'png', 'jpeg'] and contains in e and path_contains in full_path:
            results.append(full_path)
        elif os.path.isdir(full_path):
            results.extend(list_img_files_recursively(full_path, contains=contains, path_contains=path_contains))
    return results

if __name__ == '__main__':
    dl = KittiModule().test_dataloader()
    sample = next(iter(dl))
    print(sample)
