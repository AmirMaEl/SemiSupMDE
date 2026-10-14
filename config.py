"""Paths and default settings. Data locations can also be set with environment variables."""
import os

REPO = os.path.dirname(os.path.abspath(__file__))

# KITTI raw data root (contains 2011_09_26/, 2011_09_28/, ...)
KITTI_ROOT = os.environ.get('KITTI_ROOT', os.path.join(REPO, 'data/kitti_data'))
# Virtual KITTI 1.3.1 root (contains vkitti_1.3.1_rgb/ and vkitti_1.3.1_depthgt/)
VKITTI_ROOT = os.environ.get('VKITTI_ROOT', os.path.join(REPO, 'data/vkitti'))

CHECKPOINT_DIR = os.path.join(REPO, 'checkpoints')
PRETRAINED = os.path.join(CHECKPOINT_DIR, 'pretrained_vKITTI.pth')
FINETUNED = os.path.join(CHECKPOINT_DIR, 'finetuned_KITTI.pth')

SPLITS = {
    'test': os.path.join(REPO, 'splits/kitti_eigen_test.txt'),   # Eigen test split, 697 images
    'val': os.path.join(REPO, 'splits/kitti_eigen_val.txt'),     # Eigen validation split, 888 images
    'train': os.path.join(REPO, 'splits/kitti_train.txt'),       # unlabelled KITTI target images, 22600
    'vkitti': os.path.join(REPO, 'splits/vkitti_train.txt'),     # vKITTI source images, 17008
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
