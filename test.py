import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from dataloader.data_loader import KittiModule
import matplotlib.pyplot as plt
from tqdm import tqdm
from util import util
import numpy as np
import cv2
import os
import time
from accelerate import Accelerator
from torch.utils.data import ConcatDataset, DataLoader
from torch.optim import lr_scheduler
import random
from model import network
import argparse

# Constants
timestr = time.strftime("%Y%m%d-%H:%M:%S")


def test_kitti(model,eval_dist):
    """
    Test the model on the KITTI dataset and print evaluation metrics.
    
    Args:
        model: The pre-trained model to be tested.
    
    Returns:
        A dictionary containing the evaluation metrics.
    """
    min_depth = 1e-3
    max_depth = eval_dist
    dataset = KittiModule().test_dataloader()
    num_samples = len(dataset)
    
    # Initialize evaluation metrics arrays
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    rmse = np.zeros(num_samples, np.float32)
    rmse_log = np.zeros(num_samples, np.float32)
    a1 = np.zeros(num_samples, np.float32)
    a2 = np.zeros(num_samples, np.float32)
    a3 = np.zeros(num_samples, np.float32)
    
    model.eval()
    
    # Iterate over the test dataset
    for i, sample in enumerate(tqdm(dataset)):
        with torch.no_grad():
            predicted_depth = model(sample['image'].cuda())
            predicted_depth = predicted_depth[-1].squeeze()
            ground_depth = sample['gt'].squeeze().data.cpu().numpy()
            height, width = ground_depth.shape
            predicted_depth = cv2.resize(predicted_depth.data.cpu().numpy(), (width, height), interpolation=cv2.INTER_LINEAR)
            predicted_depth = (predicted_depth + 1) / 2 * 80
            predicted_depth[predicted_depth < min_depth] = min_depth
            predicted_depth[predicted_depth > max_depth] = max_depth
            
            mask = np.logical_and(ground_depth > min_depth, ground_depth < max_depth)
            crop = np.array([0.40810811 * height, 0.99189189 * height, 0.03594771 * width, 0.96405229 * width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
            
            abs_rel[i], sq_rel[i], rmse[i], rmse_log[i], a1[i], a2[i], a3[i] = util.compute_errors(ground_depth[mask], predicted_depth[mask])
    
    # Print evaluation metrics
    print('{:>10},{:>10},{:>10},{:>10},{:>10},{:>10},{:>10}'.format('abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3'))
    print('{:10.3f},{:10.3f},{:10.3f},{:10.3f},{:10.3f},{:10.3f},{:10.3f}'.format(abs_rel.mean(), sq_rel.mean(), rmse.mean(), rmse_log.mean(), a1.mean(), a2.mean(), a3.mean()))
    
    results = {
        'abs_rel': abs_rel.mean(),
        'sq_rel': sq_rel.mean(),
        'rmse': rmse.mean(),
        'rmse_log': rmse_log.mean(),
        'a1': a1.mean(),
        'a2': a2.mean(),
        'a3': a3.mean()
    }
    
    model.train()
    return results

def load_model(model_path):
    """
    Load the pre-trained model from the specified path.
    
    Args:
        model_path: Path to the model file.
    
    Returns:
        The loaded model in evaluation mode.
    """
    model = network.define_G(3, 1, 64, 4, 'batch', 'PReLU', 'UNet', 'kaiming', 0, False, [0, 1], 0.1)
    state_dict = torch.load(model_path)
    n = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(n)
    model.cuda()
    model.eval()
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./checkpoints/finetuned_KITTI.pth')
    parser.add_argument('--dist', type=float, default=50)
    args = parser.parse_args()
    # Uncomment to train the model
    # train()
    
    # Load the model and test on KITTI dataset
    model = load_model(args.model_path)
    test_kitti(model, args.dist)
