"""Consistency Regularisation for Unsupervised Domain Adaptation in Monocular Depth Estimation (CoLLAs 2024).

    import semisupmde
    model = semisupmde.load_model('finetuned')               # downloaded on first use
    depth = semisupmde.predict_depth(model, 'image.png')     # metres, H x W
    semisupmde.evaluate(model, 'test', caps=(50, 80))        # KITTI Eigen test split (semisupmde download kitti test)
"""
from .evaluate import evaluate, print_results
from .finetune import finetune
from .model import MAX_DEPTH, UNet, load_model, predict_depth

__version__ = '1.0.0'
__all__ = ['UNet', 'MAX_DEPTH', 'load_model', 'predict_depth', 'evaluate', 'print_results', 'finetune']
