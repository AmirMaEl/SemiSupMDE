import torch
import torch.nn.functional as F


def l1(pred, target):
    """L1 loss on depth in [-1, 1], scaled by 10."""
    return torch.mean(torch.abs(10. * target - 10. * pred))


def scale_pyramid(img, num_scales):
    """Downscale img by factors 1, 2, 4, ... and return the list ordered from coarse to fine."""
    h, w = img.shape[2:]
    scaled = [img] + [F.interpolate(img, size=(h // 2 ** i, w // 2 ** i), mode='bilinear', align_corners=True)
                      for i in range(1, num_scales)]
    return scaled[::-1]


def pairwise_source_loss(preds1, preds2, gt1, gt2):
    """
    Pairwise source loss (Eq. 4-7): L1 between the sum of the predictions for two source images and
    the sum of their ground truth depths, over the three finest output scales.
    """
    preds1, preds2 = preds1[2:], preds2[2:]
    gts1, gts2 = scale_pyramid(gt1, len(preds1)), scale_pyramid(gt2, len(preds2))
    return sum(l1(p1 + p2, g1 + g2) for p1, p2, g1, g2 in zip(preds1, preds2, gts1, gts2))


def consistency_loss(preds, preds_aug):
    """Perturbation consistency (Sec. 3.1): L1 between the predictions for an image and an augmented view."""
    return sum(l1(p, p_aug) for p, p_aug in zip(preds, preds_aug))
