# TediGAN
[![Paper](http://img.shields.io/badge/paper-arxiv.2010.04513-green.svg)](https://arxiv.org/abs/2012.03308)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/made%20with-python-green.svg?style=flat-square)](https://www.python.org/)

Implementation of *TediGAN: Text-Guided Diverse Image Generation and Manipulation* in PyTorch.

## [Preprint](https://arxiv.org/abs/2012.03308) | [Project](https://xiaweihao.com/projects/tedigan/) | [Dataset](https://github.com/weihaox/Multi-Modal-CelebA-HQ) | [Video](https://youtu.be/L8Na2f5viAM)

<p align="center">
<img src="/asserts/teaser.jpg"/>
</p>

Official repository for the paper W. Xia, Y. Yang, J.-H. Xue, and B. Wu. "Text-Guided Diverse Image Generation and Manipulation". 

Contact: weihaox AT outlook.com

> **NOTE**: The results reported in the paper are about [[faces](https://github.com/weihaox/Multi-Modal-CelebA-HQ)]. We are currently experimenting on other datasets. The codebase includes stylegan training, stylegan inversion, and visual-linguistic learning. The codes will be released when we finish the corresponding training on the new datasets.

## TediGAN Framework

We have proposed a novel method (abbreviated as *TediGAN*) for image synthesis using textual descriptions, which unifies two different tasks (text-guided image generation and manipulation) into the same framework and achieves high accessibility, diversity, controllability, and accurateness for facial image generation and manipulation. Through the proposed multi-modal GAN inversion and large-scale multi-modal dataset, our method can effectively synthesize images with unprecedented quality. 

<p align="center">
<img src="/asserts/control_mechanism.jpg"/>
</p>

### Train

#### Train the StyleGAN Generator

We use the training scripts from [genforce](https://github.com/genforce/genforce). You should prepare the required dataset to train StyleGAN generator ([FFHQ](https://github.com/NVlabs/ffhq-dataset) for faces or [LSUN](https://github.com/fyu/lsun) Bird for birds).

- Train on FFHQ dataset:
`
GPUS=8 
CONFIG=configs/stylegan_ffhq256.py
WORK_DIR=work_dirs/stylegan_ffhq256_train
./scripts/dist_train.sh ${GPUS} ${CONFIG} ${WORK_DIR}
`

- Train on LSUN Bird dataset:
`
GPUS=8 
CONFIG=configs/stylegan_lsun_bird256.py
WORK_DIR=work_dirs/stylegan_lsun_bird256_train
./scripts/dist_train.sh ${GPUS} ${CONFIG} ${WORK_DIR}
`

Or you can directly use a pretrained StyleGAN generator for [ffhq_face_1024](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EdfMxgb0hU9BoXwiR3dqYDEBowCSEF1IcsW3n4kwfoZ9OQ?e=VwIV58&download=1), [ffhq_face_256](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/ES-NAUCC2qdHg87BftvlBiQBVpbJ8-005Q4TNr5KrOxQEw?e=00AnWt&download=1), [cub_bird_256](), or [lsun_bird_256]().

#### Train the Image Encoder

TBD

#### Train the Text Encoder

TBD

### More Results

<p align="center">
<img src="/asserts/results/high-res-gene.png"/> 
<i>a smiling young woman with short blonde hair</i>
</p>
<p align="center">
<img src="/asserts/results/high-res-lab.png"/>
<i>he is young and wears beard</i>
</p>
<p align="center">
<img src="/asserts/results/high-res-skt.png"/> 
<i>a young woman with long black hair</i>
</p>

## Text-to-image Benchmark

### Datasets

- Multi-Modal-CelebA-HQ Dataset [[Link](https://github.com/weihaox/Multi-Modal-CelebA-HQ)]
- CUB Bird Dataset [[Link](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)]
- COCO Dataset [[Link](http://cocodataset.org)]

### Publications

Below is a curated list of related publications with codes.

#### Text-to-image Generation

- <a name="DF-GAN"></a> **[DF-GAN]** Deep Fusion Generative Adversarial Networks for Text-to-Image Synthesis (**2020**) [[paper](https://arxiv.org/pdf/2008.05865)] [[code](https://github.com/tobran/DF-GAN)]
- <a name="ControlGAN"></a> **[ControlGAN]** Controllable Text-to-Image Generation (**NeurIPS 2019**) [[paper](https://papers.nips.cc/paper/8480-controllable-text-to-image-generation.pdf)] [[code](https://github.com/mrlibw/ControlGAN)]
- <a name="DM-GAN"></a> **[DM-GAN]** Dynamic Memory Generative Adversarial Networks for Text-to-Image Synthesis (**CVPR 2019**) [[paper](https://arxiv.org/abs/1904.01310)] [[code](https://github.com/MinfengZhu/DM-GAN)]
- <a name="MirrorGAN"></a> **[MirrorGAN]** Learning Text-to-image Generation by Redescription (**CVPR 2019**) [[paper](https://arxiv.org/abs/1903.05854)] [[code](https://github.com/qiaott/MirrorGAN)]
- <a name=""></a>**[Obj-GAN]** Object-driven Text-to-Image Synthesis via Adversarial Training **(CVPR 2019)** [[paper](https://arxiv.org/abs/1902.10740)] [[code](https://github.com/jamesli1618/Obj-GAN)]
- <a name="SD-GAN"></a> **[SD-GAN]** Semantics Disentangling for Text-to-Image Generation **(CVPR 2019)** [[paper](https://arxiv.org/abs/1904.01480)] [[code](https://github.com/gjyin91/SDGAN)]
- <a name="HD-GAN"></a> **[HD-GAN]** Photographic Text-to-Image Synthesis with a Hierarchically-nested Adversarial Network (**CVPR 2018**) [[paper](https://arxiv.org/pdf/1802.09178.pdf)] [[code](https://github.com/ypxie/HDGan)]
- <a name="AttnGAN"></a> **[AttnGAN]** Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks (**CVPR 2018**) [[paper](https://arxiv.org/abs/1711.10485)] [[code](https://github.com/taoxugit/AttnGAN)]
- <a name="StackGAN++"></a> **[StackGAN++]** Realistic Image Synthesis with Stacked Generative Adversarial Networks (**TPAMI 2018**) [[paper](https://github.com/hanzhanggit/StackGAN-v2)] [[code](https://github.com/hanzhanggit/StackGAN-v2)]
- <a name="StackGAN"></a> **[StackGAN]** Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks (**ICCV 2017**) [[paper](https://arxiv.org/abs/1710.10916)] [[code](https://github.com/hanzhanggit/StackGAN)]
- <a name="GAN-INT-CLS"></a> **[GAN-INT-CLS]** Generative Adversarial Text to Image Synthesis (**ICML 2016**) [[paper](https://arxiv.org/abs/1605.05396)] [[code](https://github.com/reedscot/icml2016)]

#### Text-guided Image Manipulation

- <a name="ManiGAN"></a> **[ManiGAN]** ManiGAN: Text-Guided Image Manipulation
 (**CVPR 2020**) [[paper](https://arxiv.org/abs/1912.06203)] [[code](https://github.com/mrlibw/ManiGAN)]
- <a name="Lightweight-Manipulation"></a> **[Lightweight-Manipulation]** Lightweight Generative Adversarial Networks for Text-Guided Image Manipulation (**NeurIPS 2020**) [[paper](https://arxiv.org/abs/2010.12136)] [[code](https://github.com/mrlibw/Lightweight-Manipulation)]
- <a name="SISGAN"></a> **[SISGAN]** Semantic Image Synthesis via Adversarial Learning (**ICCV 2017**) [[paper](https://arxiv.org/abs/1707.06873)] [[code](https://github.com/woozzu/dong_iccv_2017)]
- <a name="TAGAN"></a> **[TAGAN]** Text-Adaptive Generative Adversarial Networks: Manipulating Images with Natural Language (**NeurIPS 2018**) [[paper](https://arxiv.org/abs/1810.11919)] [[code](https://github.com/woozzu/tagan)]

### Metrics

- FID ([[paper](https://arxiv.org/abs/1706.08500)] [[code](https://github.com/bioinf-jku/TTUR)] )
- Inception-Score ([[paper](https://arxiv.org/abs/1606.03498)] [[code](https://github.com/hanzhanggit/StackGAN-inception-model)])
- LIPIPS ([[paper](https://arxiv.org/abs/1801.03924)] [[code](https://www.github.com/richzhang/PerceptualSimilarity)])

## Citation

If you find our work, code or the benchmark helpful for your research, please consider to cite:

```
@article{xia2020tedigan,
  title={TediGAN: Text-Guided Diverse Face Image Generation and Manipulation},
  author={Xia, Weihao and Yang, Yujiu and Xue, Jing-Hao and Wu, Baoyuan},
  journal={arXiv preprint arXiv: 2012.03308},
  year={2020}
}
```
## Acknowledgments

Code borrows heavily from [idinvert](https://github.com/genforce/idinvert_pytorch) and [genforce](https://github.com/genforce/genforce).