"""RandAugment with the FixMatch augmentation pool and a fixed CutOut (App. A.4 of the paper).

Adapted from https://github.com/kekmodel/FixMatch-pytorch, which is based on
https://github.com/google-research/fixmatch/blob/master/third_party/auto_augment/augmentations.py
"""
import random

import numpy as np
import PIL
import PIL.ImageDraw
import PIL.ImageEnhance
import PIL.ImageOps

PARAMETER_MAX = 10


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def _signed(v):
    return -v if random.random() < 0.5 else v


def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    return PIL.ImageEnhance.Brightness(img).enhance(_float_parameter(v, max_v) + bias)


def Color(img, v, max_v, bias=0):
    return PIL.ImageEnhance.Color(img).enhance(_float_parameter(v, max_v) + bias)


def Contrast(img, v, max_v, bias=0):
    return PIL.ImageEnhance.Contrast(img).enhance(_float_parameter(v, max_v) + bias)


def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)


def Identity(img, **kwarg):
    return img


def Posterize(img, v, max_v, bias=0):
    return PIL.ImageOps.posterize(img, _int_parameter(v, max_v) + bias)


def Rotate(img, v, max_v, bias=0):
    return img.rotate(_signed(_int_parameter(v, max_v) + bias))


def Sharpness(img, v, max_v, bias=0):
    return PIL.ImageEnhance.Sharpness(img).enhance(_float_parameter(v, max_v) + bias)


def ShearX(img, v, max_v, bias=0):
    v = _signed(_float_parameter(v, max_v) + bias)
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, max_v, bias=0):
    v = _signed(_float_parameter(v, max_v) + bias)
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(img, v, max_v, bias=0):
    return PIL.ImageOps.solarize(img, 256 - (_int_parameter(v, max_v) + bias))


def TranslateX(img, v, max_v, bias=0):
    v = int(_signed(_float_parameter(v, max_v) + bias) * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v, max_v, bias=0):
    v = int(_signed(_float_parameter(v, max_v) + bias) * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def CutoutAbs(img, v):
    """Paint a grey v x v square at a random position."""
    w, h = img.size
    x0 = int(max(0, np.random.uniform(0, w) - v / 2.))
    y0 = int(max(0, np.random.uniform(0, h) - v / 2.))
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle((x0, y0, int(min(w, x0 + v)), int(min(h, y0 + v))), (127, 127, 127))
    return img


FIXMATCH_POOL = [(AutoContrast, None, None),
                 (Brightness, 0.9, 0.05),
                 (Color, 0.9, 0.05),
                 (Contrast, 0.9, 0.05),
                 (Equalize, None, None),
                 (Identity, None, None),
                 (Posterize, 4, 4),
                 (Rotate, 30, 0),
                 (Sharpness, 0.9, 0.05),
                 (ShearX, 0.3, 0),
                 (ShearY, 0.3, 0),
                 (Solarize, 256, 0),
                 (TranslateX, 0.3, 0),
                 (TranslateY, 0.3, 0)]


class RandAugment:
    """
    Apply n random operations from the FixMatch pool (each with probability 0.5 and a random
    magnitude below m), followed by a 16x16 CutOut.
    """
    def __init__(self, n, m):
        assert n >= 1 and 1 <= m <= 10
        self.n, self.m = n, m

    def __call__(self, img):
        for op, max_v, bias in random.choices(FIXMATCH_POOL, k=self.n):
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        return CutoutAbs(img, 16)
