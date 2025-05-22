# Logit Distillation via Global Distribution Alignment
#### This repo provides the source code & data of our paper: Logit Distillation via Global Distribution Alignment.

## Dependencies
* conda create -n gda python=3.7 -y
* torch==1.11.0+cu113
* torchvision==0.12.0+cu113
* torchaudio==0.11.0+cu113
* timm==0.6.12

## Image Classification
### Preparation
1. Download [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) images from its website.
2. Put the dataset into **Image Classification\cache\data\cifar**.
3. Download the pre-trained weights from [Strong-to-Weak](https://github.com/megvii-research/mdistiller/releases/tag/checkpoints) and [Weak-to-Strong](https://github.com/ggjy/vision_weak_to_strong/releases/tag/cifar-ckpt-1).
4. Put weights into **Image Classification\cache\ckpt\cifar**.


