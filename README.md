# CONSISTENCY REGULARISATION FOR UNSUPERVISED DOMAIN ADAPTATION IN MONOCULAR DEPTH ESTIMATION
Amir El-Ghoussani, Julia Hornauer, Gustavo Carneiro, Vasileios Belagiannis

This is the official codebase for the paper [Consistency Regularisation for Unsupervised Domain Adaptation in Monocular Depth Estimation](https://arxiv.org/abs/2405.17704) (CoLLAs 2024). It contains the vKITTI → KITTI setting: evaluation on the KITTI Eigen test split and the domain-adaptation finetuning.
![overview of proposed finetuning approach](https://arxiv.org/html/2405.17704v1/extracted/5624780/overview.png "overview")
# Abstract
In monocular depth estimation, unsupervised domain adaptation has recently been explored to relax the dependence on large annotated image-based depth datasets. However, this comes at the cost of training multiple models or requiring complex training protocols. We formulate unsupervised domain adaptation for monocular depth estimation as a consistency-based semi-supervised learning problem by assuming access only to the source domain ground truth labels. To this end, we introduce a pairwise loss function that regularises predictions on the source domain while enforcing perturbation consistency across multiple augmented views of the unlabelled target samples. Importantly, our approach is simple and effective, requiring only training of a single model in contrast to the prior work. In our experiments, we rely on the standard depth estimation benchmarks KITTI and NYUv2 to demonstrate state-of-the-art results compared to related approaches. Furthermore, we analyse the simplicity and effectiveness of our approach in a series of ablation studies.

# Getting started
One command installs the environment, downloads the models and the test data, and evaluates the finetuned model (Linux, CUDA GPU, Python >= 3.8, about 4 GB of disk):
```
git clone https://github.com/AmirMaEl/SemiSupMDE.git && cd SemiSupMDE && ./run.sh
```
or, without cloning first: `curl -sSL https://raw.githubusercontent.com/AmirMaEl/SemiSupMDE/main/run.sh | bash`

| Command | What it does |
|---|---|
| `./run.sh` | evaluate `finetuned_KITTI.pth` on the KITTI Eigen test split |
| `./run.sh eval <model.pth>` | evaluate another checkpoint, e.g. `checkpoints/pretrained_vKITTI.pth` |
| `./run.sh finetune [args]` | vKITTI → KITTI finetuning (see below) |
| `./run.sh setup` | only create the environment and download the models |

`run.sh` creates a virtual environment in `.venv/` (with [uv](https://docs.astral.sh/uv/) if it is installed, otherwise `python3 -m venv`) and installs `requirements.txt`. Every step is skipped when its output already exists, so the command can be rerun or resumed after an interruption. The downloads are done by `download.py`:
- `python download.py models`: the two checkpoints into `./checkpoints`
    - `pretrained_vKITTI.pth`: pretrained on vKITTI only (starting point of the finetuning)
    - `finetuned_KITTI.pth`: after domain adaptation to KITTI (the model reported in the paper)
- `python download.py kitti test val train`: the KITTI raw files of the given splits into `KITTI_ROOT` (default `./data/kitti_data`). Only the files the splits use are read out of the drive archives on the KITTI server with HTTP range requests: 1.8 GB for the test split instead of 76 GB of full drives.
- `python download.py vkitti`: [Virtual KITTI 1.3.1](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-1/) RGB and depth into `VKITTI_ROOT` (default `./data/vkitti`, 20 GB, only needed for finetuning)

To use data you already have, set `KITTI_ROOT` (containing `2011_09_26/`, ...) and `VKITTI_ROOT` (containing `vkitti_1.3.1_rgb/` and `vkitti_1.3.1_depthgt/`) before running. By downloading the data you agree to the licenses of [KITTI](https://www.cvlibs.net/datasets/kitti/raw_data.php) and [Virtual KITTI](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-1/).

# Evaluation
```
./run.sh eval checkpoints/finetuned_KITTI.pth     # or: python evaluate.py --model_path checkpoints/finetuned_KITTI.pth
```
Evaluates on the 697 images of the KITTI Eigen test split with the Garg crop, ground truth from the projected velodyne points and depth capped at 50 m and 80 m (no median scaling). Expected results:

| Model | Cap | AbsRel | SqRel | RMSE | RMSE log | δ<1.25 | δ<1.25² | δ<1.25³ |
|---|---|---|---|---|---|---|---|---|
| `pretrained_vKITTI.pth` | 50 m | 0.170 | 1.372 | 4.991 | 0.262 | 0.769 | 0.907 | 0.957 |
| `finetuned_KITTI.pth` | 50 m | 0.161 | 1.029 | 4.449 | 0.237 | 0.785 | 0.918 | 0.968 |
| `finetuned_KITTI.pth` | 80 m | 0.168 | 1.347 | 6.002 | 0.254 | 0.769 | 0.907 | 0.961 |

# Finetuning
```
./run.sh finetune          # or: python finetune.py
```
This downloads the remaining KITTI images (Eigen validation split and the unlabelled target images, 17 GB more) and Virtual KITTI (20 GB); about 25 min on a fast connection. Starting from `pretrained_vKITTI.pth`, each step jointly optimises the pairwise source loss on 12 vKITTI pairs and the consistency loss between 6 unlabelled KITTI images and two RandAugment views of each, in one forward pass over 42 images (about 35 GB of GPU memory). The defaults in `config.py` are the settings of the paper: Adam with a constant learning rate of 4e-8, 941 steps per epoch, 10 epochs (about 2.5 h on an A40). After every epoch the model is evaluated on the Eigen validation and test splits; `best_val.pth` is selected on the validation split. Checkpoints and `metrics.jsonl` are written to `./checkpoints/finetune_<time>/`.

Note: the results in the paper were obtained with gradients accumulated over all steps, i.e. without resetting `.grad` between optimiser steps. This is the default (`--accumulate_grads 1`). With `--accumulate_grads 0` Adam's steps at this learning rate are several times smaller and the model barely adapts.

# Code structure
| File | Content |
|---|---|
| `run.sh` | single entry point: environment, downloads, evaluation and finetuning |
| `download.py` | model and data downloads |
| `config.py` | data paths, splits and finetuning settings |
| `model.py` | U-Net depth network |
| `loaders/kitti.py` | KITTI evaluation data (velodyne ground truth) and unlabelled target images |
| `loaders/vkitti.py` | vKITTI source pairs |
| `loaders/rand_augment.py` | RandAugment (FixMatch pool + CutOut) for the target views |
| `losses.py` | pairwise source loss and consistency loss |
| `evaluate.py` | KITTI Eigen evaluation |
| `finetune.py` | domain-adaptation finetuning |

# Cite our work
```
@misc{elghoussani2024consistencyregularisationunsuperviseddomain,
      title={Consistency Regularisation for Unsupervised Domain Adaptation in Monocular Depth Estimation},
      author={Amir El-Ghoussani and Julia Hornauer and Gustavo Carneiro and Vasileios Belagiannis},
      year={2024},
      eprint={2405.17704},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2405.17704},
}
```
