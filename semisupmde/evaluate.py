"""Evaluate a depth model on the KITTI Eigen split (Garg crop, depth capped at 50 m or 80 m)."""
import cv2
import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm

from . import config
from .data.kitti import KittiDepth
from .model import MAX_DEPTH

MIN_DEPTH = 1e-3
METRICS = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']


def compute_errors(gt, pred):
    thresh = np.maximum(gt / pred, pred / gt)
    return dict(abs_rel=np.mean(np.abs(gt - pred) / gt),
                sq_rel=np.mean((gt - pred) ** 2 / gt),
                rmse=np.sqrt(np.mean((gt - pred) ** 2)),
                rmse_log=np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2)),
                a1=(thresh < 1.25).mean(), a2=(thresh < 1.25 ** 2).mean(), a3=(thresh < 1.25 ** 3).mean())


@torch.no_grad()
def evaluate(model, split='test', caps=(50,), kitti_root=None, num_workers=8):
    """
    Evaluate on a KITTI split. Predictions are upsampled to the ground truth resolution, clamped to
    [1e-3, cap] and compared inside the Garg crop where 1e-3 < gt < cap. No median scaling.

    Returns:
        {cap: {metric: value}}, metrics averaged over images.
    """
    was_training = model.training
    device = next(model.parameters()).device
    model.eval()
    loader = data.DataLoader(KittiDepth(config.SPLITS[split], kitti_root or config.KITTI_ROOT, config.IMG_SIZE),
                             batch_size=1, shuffle=False, num_workers=num_workers)
    errors = {cap: [] for cap in caps}
    for sample in tqdm(loader, desc='eval %s' % split, leave=False):
        gt = sample['gt'][0].numpy()
        h, w = gt.shape
        pred = model(sample['image'].to(device))[-1][0, 0].cpu().numpy()
        pred = (cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR) + 1) / 2 * MAX_DEPTH
        crop = np.zeros_like(gt, dtype=bool)
        crop[int(0.40810811 * h):int(0.99189189 * h), int(0.03594771 * w):int(0.96405229 * w)] = True
        for cap in caps:
            mask = (gt > MIN_DEPTH) & (gt < cap) & crop
            errors[cap].append(compute_errors(gt[mask], np.clip(pred, MIN_DEPTH, cap)[mask]))
    model.train(was_training)
    return {cap: {k: float(np.mean([e[k] for e in errors[cap]])) for k in METRICS} for cap in caps}


def print_results(results, title=''):
    for cap, m in results.items():
        print('%s cap %dm' % (title, cap))
        print(''.join('%10s' % k for k in METRICS))
        print(''.join('%10.3f' % m[k] for k in METRICS))
