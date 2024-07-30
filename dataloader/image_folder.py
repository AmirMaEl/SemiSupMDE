import os
import os.path
from params import KITTI_ROOT

# Supported image file extensions
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    """
    Check if a file is an image.
    
    Args:
        filename: Name of the file.
    
    Returns:
        Boolean indicating if the file is an image.
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(path_files):
    """
    Create a dataset from a directory or a text file listing image paths.
    
    Args:
        path_files: Path to the directory or text file.
    
    Returns:
        A tuple containing a list of image paths and the number of images.
    """
    if path_files.find('.txt') != -1:
        paths, size = make_dataset_txt(path_files)  
    else:
        paths, size = make_dataset_dir(path_files)

    return paths, size

def make_dataset_txt(path_files):
    """
    Create a dataset from a text file listing image paths.
    
    Args:
        path_files: Path to the text file.
    
    Returns:
        A tuple containing a list of image paths and the number of images.
    """
    image_paths = []

    with open(path_files) as f:
        paths = f.readlines()

    for path in paths:
        path = path.strip()
        path = path.replace('data/kitti_data', '')
        tmp = KITTI_ROOT + path.replace('./', '')

        if 'image_02' in tmp:
            image_paths.append(tmp)

    return image_paths, len(image_paths)


def make_dataset_dir(dir):
    """
    Create a dataset from a directory of images.
    
    Args:
        dir: Path to the directory.
    
    Returns:
        A tuple containing a list of image paths and the number of images.
    """
    image_paths = []

    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in os.walk(dir):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                image_paths.append(path)

    return image_paths, len(image_paths)
