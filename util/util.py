import numpy as np
import os
import torch
import matplotlib as mpl
import matplotlib.cm as cm
from PIL import Image

def compute_errors(ground_truth, prediction):
    """
    Compute error metrics between the ground truth and the predicted depth maps.
    
    Args:
        ground_truth: Ground truth depth map.
        prediction: Predicted depth map.
    
    Returns:
        Tuple containing absolute relative difference, squared relative difference,
        root mean squared error, root mean squared error of log, and accuracy metrics (a1, a2, a3).
    """
    # Accuracy
    threshold = np.maximum((ground_truth / prediction), (prediction / ground_truth))
    a1 = (threshold < 1.25).mean()
    a2 = (threshold < 1.25 ** 2).mean()
    a3 = (threshold < 1.25 ** 3).mean()

    # Root Mean Squared Error (RMSE)
    rmse = (ground_truth - prediction) ** 2
    rmse = np.sqrt(rmse.mean())

    # RMSE (log)
    rmse_log = (np.log(ground_truth) - np.log(prediction)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    # Absolute Relative difference
    abs_rel = np.mean(np.abs(ground_truth - prediction) / ground_truth)

    # Squared Relative difference
    sq_rel = np.mean(((ground_truth - prediction) ** 2) / ground_truth)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def tensor2im(image_tensor, bytes=255.0, imtype=np.uint8):
    """
    Convert a tensor into a numpy array.
    
    Args:
        image_tensor: Input image tensor.
        bytes: Byte scale (default: 255.0).
        imtype: Desired type of the output numpy array (default: np.uint8).
    
    Returns:
        Numpy array representation of the image tensor.
    """
    if image_tensor.dim() == 3:
        image_numpy = image_tensor.cpu().float().numpy()
    else:
        image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * bytes

    return image_numpy.astype(imtype)

def numpy2im(image_numpy, bytes=255.0, imtype=np.uint8):
    """
    Convert a numpy array into an image.
    
    Args:
        image_numpy: Input numpy array.
        bytes: Byte scale (default: 255.0).
        imtype: Desired type of the output numpy array (default: np.uint8).
    
    Returns:
        Numpy array representation of the image.
    """
    image_numpy = (image_numpy * 0.5 + 0.5) * bytes

    return image_numpy.astype(imtype)

def mkdirs(paths):
    """
    Create directories if they do not exist.
    
    Args:
        paths: List of directory paths or a single directory path.
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    """
    Create a directory if it does not exist.
    
    Args:
        path: Directory path.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def get_crop_mask(size, height=192, width=640):
    """
    Generate a crop mask for an image.
    
    Args:
        size: Size of the mask.
        height: Height of the image (default: 192).
        width: Width of the image (default: 640).
    
    Returns:
        Crop mask as a tensor.
    """
    crop = np.array([0.40810811 * height,  0.99189189 * height, 0.03594771 * width, 0.96405229 * width]).astype(np.int32)
    crop_mask = np.zeros(size)
    crop_mask[:, :, crop[0]:crop[1], crop[2]:crop[3]] = 1
    if torch.cuda.is_available():
        crop_mask = torch.from_numpy(crop_mask).cuda()
    else:
        crop_mask = torch.from_numpy(crop_mask)

    return crop_mask

def convert_array_vis(depth):
    """
    Convert a depth array to a colormapped image for visualization.
    
    Args:
        depth: Input depth array.
    
    Returns:
        Colormapped image.
    """
    mask = depth != 0
    depth = depth.clip(1, 80)
    disparity = 1 / depth
    vmax = np.percentile(disparity[mask], 95)
    normalizer = mpl.colors.Normalize(vmin=disparity.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    mask = np.repeat(np.expand_dims(mask, -1), 3, -1)
    colormapped_im = (mapper.to_rgba(disparity)[:, :, :3] * 255).astype(np.uint8)
    colormapped_im[~mask] = 255
    return colormapped_im

def save_img(path, result_img):
    """
    Save an image to the specified path.
    
    Args:
        path: Path to save the image.
        result_img: Image to be saved.
    """
    pred_img1 = Image.fromarray((result_img).astype('uint8'))
    pred_img1.save(path)
