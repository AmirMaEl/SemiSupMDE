"""KITTI (raw) datasets: labelled evaluation splits and unlabelled target images for finetuning."""
import os
from collections import Counter

import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from loaders.rand_augment import RandAugment


def read_split(split_file, root):
    """Read a split file with one image path per line, relative to the KITTI root."""
    with open(split_file) as f:
        return [os.path.join(root, line.strip()) for line in f if line.strip()]


def to_tensor(img):
    """PIL image -> tensor in [-1, 1]."""
    return transforms.ToTensor()(img) * 2 - 1


def read_calib_file(path):
    data = {}
    with open(path) as f:
        for line in f:
            key, value = line.split(':', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


def velodyne_depth(image_path, im_shape, cam=2):
    """
    Project the velodyne points of a KITTI frame into the image to get the sparse ground truth depth
    (standard Eigen-split evaluation, as in Godard et al. / T2Net).
    """
    calib_dir = '/'.join(image_path.split('/')[:-4])
    velo_file = image_path.replace('image_0%d' % cam, 'velodyne_points').replace('.png', '.bin')

    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0%d' % cam].reshape(3, 4)
    P_velo2im = P_rect @ R_cam2rect @ velo2cam

    # load points, remove those behind the camera and project them into the image
    velo = np.fromfile(velo_file, dtype=np.float32).reshape(-1, 4)
    velo[:, 3] = 1.0
    velo = velo[velo[:, 0] >= 0, :]
    pts = (P_velo2im @ velo.T).T
    pts[:, :2] = pts[:, :2] / pts[:, 2][..., np.newaxis]
    pts[:, 0] = np.round(pts[:, 0]) - 1
    pts[:, 1] = np.round(pts[:, 1]) - 1
    valid = (pts[:, 0] >= 0) & (pts[:, 1] >= 0) & (pts[:, 0] < im_shape[1]) & (pts[:, 1] < im_shape[0])
    pts = pts[valid, :]

    depth = np.zeros(im_shape)
    depth[pts[:, 1].astype(int), pts[:, 0].astype(int)] = pts[:, 2]
    # where several points fall on the same pixel, keep the closest one
    inds = pts[:, 1] * (im_shape[1] - 1) + pts[:, 0] - 1
    for dupe in [item for item, count in Counter(inds).items() if count > 1]:
        p = np.where(inds == dupe)[0]
        depth[int(pts[p[0], 1]), int(pts[p[0], 0])] = pts[p, 2].min()
    depth[depth < 0] = 0
    return depth


class KittiDepth(data.Dataset):
    """
    Labelled KITTI images for evaluation: the network input (resized to img_size) and the
    velodyne ground truth at the original image resolution.
    """
    def __init__(self, split_file, root, img_size=(640, 192)):
        self.paths = read_split(split_file, root)
        self.img_size = img_size

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert('RGB')
        gt = velodyne_depth(self.paths[index], (img.size[1], img.size[0]))
        return {'image': to_tensor(img.resize(self.img_size, Image.BICUBIC)), 'gt': gt}

    def __len__(self):
        return len(self.paths)


class KittiTarget(data.Dataset):
    """
    Unlabelled KITTI target images for finetuning: x_u and two independent RandAugment views (Eq. 2-3).
    """
    def __init__(self, split_file, root, camera='image_03', randaug_n=1, randaug_m=7, img_size=(640, 192)):
        self.paths = [p.replace('image_02', camera) for p in read_split(split_file, root)]
        self.randaug = RandAugment(randaug_n, randaug_m)
        self.img_size = img_size

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert('RGB').resize(self.img_size, Image.BICUBIC)
        return {'image': to_tensor(img), 'aug1': to_tensor(self.randaug(img)), 'aug2': to_tensor(self.randaug(img))}

    def __len__(self):
        return len(self.paths)
