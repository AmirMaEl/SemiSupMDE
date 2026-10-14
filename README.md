# CONSISTENCY REGULARISATION FOR UNSUPERVISED DOMAIN ADAPTATION IN MONOCULAR DEPTH ESTIMATION
Amir El-Ghoussani, Julia Hornauer, Gustavo Carneiro, Vasileios Belagiannis

[![Paper](https://img.shields.io/badge/PMLR-CoLLAs%202024-blue)](https://proceedings.mlr.press/v274/el-ghoussani25a.html)
[![arXiv](https://img.shields.io/badge/arXiv-2405.17704-b31b1b)](https://arxiv.org/abs/2405.17704)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-models-yellow)](https://huggingface.co/AmirMaEl/SemiSupMDE)

This is the official codebase for the paper [Consistency Regularisation for Unsupervised Domain Adaptation in Monocular Depth Estimation](https://arxiv.org/abs/2405.17704) (CoLLAs 2024). It contains the vKITTI → KITTI setting: evaluation on the KITTI Eigen test split and the domain-adaptation finetuning.
![overview of proposed finetuning approach](https://arxiv.org/html/2405.17704v1/extracted/5624780/overview.png "overview")
# Abstract
In monocular depth estimation, unsupervised domain adaptation has recently been explored to relax the dependence on large annotated image-based depth datasets. However, this comes at the cost of training multiple models or requiring complex training protocols. We formulate unsupervised domain adaptation for monocular depth estimation as a consistency-based semi-supervised learning problem by assuming access only to the source domain ground truth labels. To this end, we introduce a pairwise loss function that regularises predictions on the source domain while enforcing perturbation consistency across multiple augmented views of the unlabelled target samples. Importantly, our approach is simple and effective, requiring only training of a single model in contrast to the prior work. In our experiments, we rely on the standard depth estimation benchmarks KITTI and NYUv2 to demonstrate state-of-the-art results compared to related approaches. Furthermore, we analyse the simplicity and effectiveness of our approach in a series of ablation studies.

# Models
The checkpoints are on Hugging Face: [AmirMaEl/SemiSupMDE](https://huggingface.co/AmirMaEl/SemiSupMDE).

| Model | Description | AbsRel (KITTI, 50 m) | Download |
|---|---|---|---|
| `finetuned` | after domain adaptation vKITTI → KITTI, the model reported in the paper | 0.161 | [finetuned_KITTI.pth](https://huggingface.co/AmirMaEl/SemiSupMDE/resolve/main/finetuned_KITTI.pth) |
| `pretrained` | pretrained on Virtual KITTI only, the starting point of the adaptation | 0.170 | [pretrained_vKITTI.pth](https://huggingface.co/AmirMaEl/SemiSupMDE/resolve/main/pretrained_vKITTI.pth) |

You don't need to download them by hand: `semisupmde.load_model('finetuned')` and all `run.sh` commands fetch them on first use (with the FAUbox links as a fallback) and check them against their md5. To load a checkpoint without this package, see the [model card](https://huggingface.co/AmirMaEl/SemiSupMDE), which has a standalone `model.py`.

# Getting started
One command installs everything, downloads the models and the test data, and evaluates the finetuned model (Linux, Python >= 3.8, about 4 GB of disk; a CUDA GPU is recommended):
```
git clone https://github.com/AmirMaEl/SemiSupMDE.git && cd SemiSupMDE && ./run.sh
```
or, without cloning first: `curl -sSL https://raw.githubusercontent.com/AmirMaEl/SemiSupMDE/main/run.sh | bash`

| Command | What it does |
|---|---|
| `./run.sh` | evaluate the finetuned model on the KITTI Eigen test split |
| `./run.sh eval pretrained` | evaluate another model: `finetuned`, `pretrained` or a checkpoint path |
| `./run.sh predict img1.png img2.jpg` | predict depth for your own images (`<name>_depth.png` colourised, `<name>_depth.npy` in metres) |
| `./run.sh finetune [--lr 4e-8 ...]` | vKITTI → KITTI finetuning (see below) |
| `./run.sh setup` | only install and download the models |

`run.sh` creates a virtual environment in `.venv/` (with [uv](https://docs.astral.sh/uv/) if it is installed, otherwise `python3 -m venv`), installs PyTorch built for the installed NVIDIA driver and then this package. Every step is skipped when its output already exists, so the command can be rerun or resumed after an interruption. Models go to `./checkpoints`, data to `./data`.

# Using it as a library
```
pip install git+https://github.com/AmirMaEl/SemiSupMDE.git
```
```python
import semisupmde

model = semisupmde.load_model('finetuned')            # or 'pretrained', or a checkpoint path; downloaded on first use
depth = semisupmde.predict_depth(model, 'image.png')  # metres, at the resolution of the input (path, PIL image or array)

results = semisupmde.evaluate(model, 'test', caps=(50, 80))   # KITTI Eigen test split, after `semisupmde download kitti test`
semisupmde.print_results(results)

run_dir = semisupmde.finetune(init='pretrained', epochs=10)   # any setting of config.FINETUNE can be overridden
```
The same is available on the command line as `semisupmde {download,eval,predict,finetune}` (or `python -m semisupmde`); `eval` and `finetune` download the data they need. Models and data are stored under `SEMISUPMDE_HOME` (default `~/.cache/semisupmde`); `run.sh` sets it to the repository.

# Data
`semisupmde download ...` (or `./run.sh download ...`) fetches
- `models`: the two checkpoints from [Hugging Face](https://huggingface.co/AmirMaEl/SemiSupMDE) into `$SEMISUPMDE_HOME/checkpoints` (see [Models](#models))
- `kitti test val train`: the KITTI raw files of the given splits into `KITTI_ROOT` (default `$SEMISUPMDE_HOME/data/kitti_data`). Only the files the splits use are read out of the drive archives on the KITTI server with HTTP range requests: 1.8 GB for the test split instead of 76 GB of full drives.
- `vkitti`: [Virtual KITTI 1.3.1](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-1/) RGB and depth into `VKITTI_ROOT` (default `$SEMISUPMDE_HOME/data/vkitti`, 20 GB, only needed for finetuning)

To use data you already have, set `KITTI_ROOT` (containing `2011_09_26/`, ...) and `VKITTI_ROOT` (containing `vkitti_1.3.1_rgb/` and `vkitti_1.3.1_depthgt/`). By downloading the data you agree to the licenses of [KITTI](https://www.cvlibs.net/datasets/kitti/raw_data.php) and [Virtual KITTI](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-1/).

# Evaluation
```
./run.sh eval finetuned     # or: semisupmde eval --model finetuned
```
Evaluates on the 697 images of the KITTI Eigen test split with the Garg crop, ground truth from the projected velodyne points and depth capped at 50 m and 80 m (no median scaling). Expected results:

| Model | Cap | AbsRel | SqRel | RMSE | RMSE log | δ<1.25 | δ<1.25² | δ<1.25³ |
|---|---|---|---|---|---|---|---|---|
| `pretrained_vKITTI.pth` | 50 m | 0.170 | 1.372 | 4.991 | 0.262 | 0.769 | 0.907 | 0.957 |
| `finetuned_KITTI.pth` | 50 m | 0.161 | 1.029 | 4.449 | 0.237 | 0.785 | 0.918 | 0.968 |
| `finetuned_KITTI.pth` | 80 m | 0.168 | 1.347 | 6.002 | 0.254 | 0.769 | 0.907 | 0.961 |

# Finetuning
```
./run.sh finetune     # or: semisupmde finetune
```
This downloads the remaining KITTI images (Eigen validation split and the unlabelled target images, 17 GB more) and Virtual KITTI (20 GB); about 25 min on a fast connection. Starting from `pretrained_vKITTI.pth`, each step jointly optimises the pairwise source loss on 12 vKITTI pairs and the consistency loss between 6 unlabelled KITTI images and two RandAugment views of each, in one forward pass over 42 images (about 35 GB of GPU memory). The defaults in `semisupmde/config.py` are the settings of the paper: Adam with a constant learning rate of 4e-8, 941 steps per epoch, 10 epochs (about 2.5 h on an A40). After every epoch the model is evaluated on the Eigen validation and test splits; `best_val.pth` is selected on the validation split. Checkpoints and `metrics.jsonl` are written to `checkpoints/finetune_<time>/`.

Note: the results in the paper were obtained with gradients accumulated over all steps, i.e. without resetting `.grad` between optimiser steps. This is the default (`--accumulate_grads 1`). With `--accumulate_grads 0` Adam's steps at this learning rate are several times smaller and the model barely adapts.

# Code structure
| File | Content |
|---|---|
| `run.sh` | single entry point: environment, downloads, evaluation, prediction and finetuning |
| `semisupmde/config.py` | paths, released models, splits and finetuning settings |
| `semisupmde/model.py` | U-Net depth network, `load_model`, `predict_depth` |
| `semisupmde/data/kitti.py` | KITTI evaluation data (velodyne ground truth) and unlabelled target images |
| `semisupmde/data/vkitti.py` | vKITTI source pairs |
| `semisupmde/data/rand_augment.py` | RandAugment (FixMatch pool + CutOut) for the target views |
| `semisupmde/losses.py` | pairwise source loss and consistency loss |
| `semisupmde/evaluate.py` | KITTI Eigen evaluation |
| `semisupmde/finetune.py` | domain-adaptation finetuning |
| `semisupmde/download.py` | model and data downloads |
| `semisupmde/cli.py` | `semisupmde` command line tool |
| `semisupmde/splits/` | KITTI Eigen test/val, KITTI target and vKITTI source image lists |

# Cite our work 
```
@InProceedings{pmlr-v274-el-ghoussani25a,
  title = 	 {Consistency Regularisation for Unsupervised Domain Adaptation in Monocular Depth Estimation},
  author =       {El-Ghoussani, Amir and Hornauer, Julia and Carneiro, Gustavo and Belagiannis, Vasileios},
  booktitle = 	 {Proceedings of The 3rd Conference on Lifelong Learning Agents},
  pages = 	 {577--596},
  year = 	 {2025},
  editor = 	 {Lomonaco, Vincenzo and Melacci, Stefano and Tuytelaars, Tinne and Chandar, Sarath and Pascanu, Razvan},
  volume = 	 {274},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {29 Jul--01 Aug},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v274/main/assets/el-ghoussani25a/el-ghoussani25a.pdf},
  url = 	 {https://proceedings.mlr.press/v274/el-ghoussani25a.html},
  abstract = 	 {In monocular depth estimation, unsupervised domain adaptation has recently been explored to relax the dependence on large annotated image-based depth datasets. However, this comes at the cost of training multiple models or requiring complex training protocols. We formulate unsupervised domain adaptation for monocular depth estimation as a consistency-based semi-supervised learning problem by assuming access only to the source domain ground truth labels. To this end, we introduce a pairwise loss function that regularises predictions on the source domain while enforcing perturbation consistency across multiple augmented views of the unlabelled target samples. Importantly, our approach is simple and effective, requiring only training of a single model in contrast to the prior work. In our experiments, we rely on the standard depth estimation benchmarks KITTI and NYUv2 to demonstrate state-of-the-art results compared to related approaches. Furthermore, we analyse the simplicity and effectiveness of our approach in a series of ablation studies. The code is available at \url{https://github.com/AmirMaEl/SemiSupMDE}.}
}
```
