"""U-Net depth network (T2Net-style, Zheng et al. 2018), the only model used in the paper.

Predicts depth in [-1, 1] at four scales; the last (full-resolution) output is the prediction,
converted to metres as (pred + 1) / 2 * MAX_DEPTH.
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from . import config

MAX_DEPTH = 80.0


class _InceptionBlock(nn.Module):
    def __init__(self, nc, act, width=3):
        super().__init__()
        self.width = width
        for i in range(width):
            setattr(self, 'layer' + str(i), nn.Sequential(
                nn.ReflectionPad2d(i * 2 + 1),
                nn.Conv2d(nc, nc, kernel_size=3, padding=0, dilation=i * 2 + 1, bias=False)))
        self.norm1 = nn.BatchNorm2d(nc * width)
        self.norm2 = nn.BatchNorm2d(nc)
        self.nonlinearity = act
        self.branch1x1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nc * width, nc, kernel_size=3, padding=0, bias=False))

    def forward(self, x):
        out = torch.cat([getattr(self, 'layer' + str(i))(x) for i in range(self.width)], 1)
        out = self.nonlinearity(self.norm1(out))
        out = self.norm2(self.branch1x1(out))
        return self.nonlinearity(out + x)


class _EncoderBlock(nn.Module):
    def __init__(self, input_nc, middle_nc, output_nc, act):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_nc, middle_nc, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(middle_nc), act,
            nn.Conv2d(middle_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_nc), act)

    def forward(self, x):
        return self.model(x)


class _DecoderUpBlock(nn.Module):
    def __init__(self, input_nc, middle_nc, output_nc, act):
        super().__init__()
        self.model = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc, middle_nc, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(middle_nc), act,
            nn.ConvTranspose2d(middle_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(output_nc), act)

    def forward(self, x):
        return self.model(x)


class _OutputBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.ReflectionPad2d(kernel_size // 2),
            nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, padding=0, bias=False),
            nn.Tanh())

    def forward(self, x):
        return self.model(x)


class UNet(nn.Module):
    """
    4-level U-Net with an inception bottleneck and multi-scale outputs.
    Module names match the original implementation, so released checkpoints load directly.
    As in the original, a single PReLU (one learnable slope) is shared by all layers.
    """
    def __init__(self, input_nc=3, output_nc=1, ngf=64, weight=0.1):
        super().__init__()
        self.weight = weight
        act = nn.PReLU()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf), act)
        self.conv2 = _EncoderBlock(ngf, ngf * 2, ngf * 2, act)
        self.conv3 = _EncoderBlock(ngf * 2, ngf * 4, ngf * 4, act)
        self.conv4 = _EncoderBlock(ngf * 4, ngf * 8, ngf * 8, act)
        self.center = nn.Sequential(
            *[_InceptionBlock(ngf * 8, act) for _ in range(3)],
            _DecoderUpBlock(ngf * 8, ngf * 8, ngf * 4, act))
        self.deconv4 = _DecoderUpBlock(ngf * (4 + 4), ngf * 8, ngf * 2, act)
        self.deconv3 = _DecoderUpBlock(ngf * (2 + 2) + output_nc, ngf * 4, ngf, act)
        self.deconv2 = _DecoderUpBlock(ngf * (1 + 1) + output_nc, ngf * 2, ngf // 2, act)
        self.output4 = _OutputBlock(ngf * (4 + 4), output_nc, 3)
        self.output3 = _OutputBlock(ngf * (2 + 2) + output_nc, output_nc, 3)
        self.output2 = _OutputBlock(ngf * (1 + 1) + output_nc, output_nc, 3)
        self.output1 = _OutputBlock(ngf // 2 + output_nc, output_nc, 7)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        """
        Returns:
            List [bottleneck features, 1/8, 1/4, 1/2, full-resolution prediction].
        """
        conv1 = self.pool(self.conv1(x))
        conv2 = self.pool(self.conv2(conv1))
        conv3 = self.pool(self.conv3(conv2))
        center_in = self.pool(self.conv4(conv3))
        center_out = self.center(center_in)

        skip4 = torch.cat([center_out, conv3 * self.weight], 1)
        deconv4, output4 = self.deconv4(skip4), self.output4(skip4)
        skip3 = torch.cat([deconv4, conv2 * self.weight * 0.5, self.upsample(output4)], 1)
        deconv3, output3 = self.deconv3(skip3), self.output3(skip3)
        skip2 = torch.cat([deconv3, conv1 * self.weight * 0.1, self.upsample(output3)], 1)
        deconv2, output2 = self.deconv2(skip2), self.output2(skip2)
        output1 = self.output1(torch.cat([deconv2, self.upsample(output2)], 1))
        return [center_in, output4, output3, output2, output1]


def load_model(model='finetuned', device=None):
    """
    Build the U-Net and load weights.

    Args:
        model: 'finetuned' (vKITTI -> KITTI, the model of the paper) or 'pretrained' (vKITTI only), which
            are downloaded on first use, or the path to a checkpoint (state dict, optionally saved from a
            DataParallel model).
        device: defaults to 'cuda' if available, else 'cpu'.
    """
    from .download import download_model
    path = download_model(model) if model in config.MODELS else model
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet()
    state_dict = torch.load(path, map_location='cpu')
    net.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    return net.to(device).eval()


@torch.no_grad()
def predict_depth(model, image):
    """
    Predict metric depth for one image.

    Args:
        model: a model from load_model.
        image: a file path, a PIL image or an RGB uint8 array (H x W x 3).

    Returns:
        Depth in metres as a float32 array of the input resolution (H x W).
    """
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert('RGB')
    w, h = image.size
    x = transforms.ToTensor()(image.resize(config.IMG_SIZE, Image.BICUBIC)) * 2 - 1
    device = next(model.parameters()).device
    pred = model(x[None].to(device))[-1]
    pred = torch.nn.functional.interpolate(pred, size=(h, w), mode='bilinear', align_corners=False)
    return ((pred[0, 0] + 1) / 2 * MAX_DEPTH).cpu().numpy().astype(np.float32)
