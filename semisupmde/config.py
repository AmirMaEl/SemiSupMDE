"""Paths and default settings.

Models and data live under SEMISUPMDE_HOME (default ~/.cache/semisupmde):
    checkpoints/        pretrained_vKITTI.pth, finetuned_KITTI.pth and finetuning runs
    data/kitti_data/    KITTI raw data (2011_09_26/, 2011_09_28/, ...), or set KITTI_ROOT
    data/vkitti/        Virtual KITTI 1.3.1 (vkitti_1.3.1_rgb/, vkitti_1.3.1_depthgt/), or set VKITTI_ROOT
"""
import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.abspath(os.path.expanduser(
    os.environ.get('SEMISUPMDE_HOME', os.path.join('~', '.cache', 'semisupmde'))))

KITTI_ROOT = os.environ.get('KITTI_ROOT', os.path.join(HOME, 'data', 'kitti_data'))
VKITTI_ROOT = os.environ.get('VKITTI_ROOT', os.path.join(HOME, 'data', 'vkitti'))
CHECKPOINT_DIR = os.path.join(HOME, 'checkpoints')

# released models: file name, download URL and md5
MODELS = {
    'pretrained': ('pretrained_vKITTI.pth',
                   'https://faubox.rrze.uni-erlangen.de/dl/fiRocZxYkVUdcueEYpFWWd/pretrained_vKITTI.pth',
                   'b205d459a32a6eb645efe4235372a6f5'),
    'finetuned': ('finetuned_KITTI.pth',
                  'https://faubox.rrze.uni-erlangen.de/dl/fiQirkjS1AraYBgLq9fCGx/finetuned_KITTI.pth',
                  '7af8e452fb529674c579f7d5b333db7e'),
}
PRETRAINED = os.path.join(CHECKPOINT_DIR, MODELS['pretrained'][0])
FINETUNED = os.path.join(CHECKPOINT_DIR, MODELS['finetuned'][0])

SPLITS = {
    'test': os.path.join(PACKAGE_DIR, 'splits', 'kitti_eigen_test.txt'),   # Eigen test split, 697 images
    'val': os.path.join(PACKAGE_DIR, 'splits', 'kitti_eigen_val.txt'),     # Eigen validation split, 888 images
    'train': os.path.join(PACKAGE_DIR, 'splits', 'kitti_train.txt'),       # unlabelled KITTI target images, 22600
    'vkitti': os.path.join(PACKAGE_DIR, 'splits', 'vkitti_train.txt'),     # vKITTI source images, 17008
}

IMG_SIZE = (640, 192)

# Finetuning (vKITTI -> KITTI), the settings used for the paper
FINETUNE = dict(
    lr=4e-8,               # Adam, constant learning rate
    epochs=10,
    steps_per_epoch=941,   # 22600 KITTI images / 24
    n_source=12,           # N source pairs (x_s1, x_s2) per step
    n_target=6,            # N / r target images per step, r = 2 (Eq. 15)
    source_weight=2.0,     # weight of the pairwise source loss
    randaug_n=1,           # RandAugment depth and magnitude for the target views
    randaug_m=7,
    color_jitter=0.4,      # source augmentation (plus random flips and +-5 degree rotations)
    target_camera='image_03',
    # Accumulate gradients over all steps (never reset .grad). The results in the paper were obtained
    # this way; resetting gradients every step at this learning rate barely changes the model.
    accumulate_grads=True,
)
