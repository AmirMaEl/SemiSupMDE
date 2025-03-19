# CONSISTENCY REGULARISATION FOR UNSUPERVISED DOMAIN ADAPTATION IN MONOCULAR DEPTH ESTIMATION
Amir El-Ghoussani, Julia Hornauer, Gustavo Carneiro, Vasileios Belagiannis 

  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/consistency-regularisation-for-unsupervised/unsupervised-domain-adaptation-on-virtual-2)](https://paperswithcode.com/sota/unsupervised-domain-adaptation-on-virtual-2?p=consistency-regularisation-for-unsupervised)

This is the official codebase for the paper [Consistency Regularisation for Unsupervised Domain Adaptation in Monocular Depth Estimation](https://arxiv.org/abs/2405.17704).
![overview of proposed finetuning approach](https://arxiv.org/html/2405.17704v1/extracted/5624780/overview.png "overview")
# Abstract 
In monocular depth estimation, unsupervised domain adaptation has recently been explored to relax the dependence on large annotated image-based depth datasets. However, this comes at the cost of training multiple models or requiring complex training protocols. We formulate unsupervised domain adaptation for monocular depth estimation as a consistency-based semi-supervised learning problem by assuming access only to the source domain ground truth labels. To this end, we introduce a pairwise loss function that regularises predictions on the source domain while enforcing perturbation consistency across multiple augmented views of the unlabelled target samples. Importantly, our approach is simple and effective, requiring only training of a single model in contrast to the prior work. In our experiments, we rely on the standard depth estimation benchmarks KITTI and NYUv2 to demonstrate state-of-the-art results compared to related approaches. Furthermore, we analyse the simplicity and effectiveness of our approach in a series of ablation studies. 

# Getting started
1. clone the repo.
2. install `requiremetns.txt`
3. download the pretrained models:
    - `pretrained_vKITTI.pth` 'https://faubox.rrze.uni-erlangen.de/getlink/fiRocZxYkVUdcueEYpFWWd/pretrained_vKITTI.pth'
    - `finetuned_KITTI.pth` 'https://faubox.rrze.uni-erlangen.de/getlink/fiQirkjS1AraYBgLq9fCGx/finetuned_KITTI.pth'
4. download the required data (KITTI) and put it in `/data/`, you can use the script `raw_data_downloader.sh` for that, you can also modify that `KITTI_ROOT` path in `params.py`.

<!-- 3. download the pretrained models by running `download_model.py` (puts the checkpoints under the `/checkpoints` directory)
4. download the KITTI and vKITTI data by running `download_data.py` -->
# TESTING   
After downloading the data and the models you can test the model on the KITTI eigen split using `python test.py`.
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
