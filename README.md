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
2. Put the dataset into `Image Classification/cache/data/cifar`.
3. Download the pre-trained weights from [Strong-to-Weak](https://github.com/megvii-research/mdistiller/releases/tag/checkpoints) and [Weak-to-Strong](https://github.com/ggjy/vision_weak_to_strong/releases/tag/cifar-ckpt-1).
4. Put weights into `Image Classification/cache/ckpt/cifar`.

### Training model
```python
sh train.sh
```
**Note:** 
1. We only use KD as the basis, select ResNet56 as the teacher model and ResNet20 as the student model as examples. You can freely modify the variable values ​​defined at the beginning of `train.sh`.
2. For the definitions of different distillation losses, you can find them in `Image Classification/distillers`.
3. You will see the logs in floder `logs`.

## Few-shot Learning 
### Preparation
1. Download the [miniImageNet](https://github.com/gidariss/FewShotWithoutForgetting) datasets and link the folders into `Few-shot Learning/materials` with names `mini-imagenet`.
2. You can setting the dataset path and output path in `Few-shot Learning/init_env.py`.
3. When running python programs, use --gpu to specify the GPUs for running the code (e.g. `--gpu 0,1`). For Classifier-Baseline, we train with 4 GPUs on miniImageNet. Meta-Baseline uses half of the GPUs correspondingly.

### Training model
#### Training Classifier-Baseline
* Training
```
python train_classifier.py --config config/classifier/train_classifier_mini.yaml --res_type resnet12_bottle --gpu 0,1,2,3
python train_classifier.py --config config/classifier/train_classifier_mini.yaml --res_type resnet18_bottle --gpu 0,1,2,3
python train_classifier.py --config config/classifier/train_classifier_mini.yaml --res_type resnet36_bottle --gpu 0,1,2,3
```
* Knowledge Distillation
```
python train_classifier.py --config config/classifier/train_classifier_mini_kd.yaml --res_type resnet36_bottle --teacher_res_type resnet12_bottle --gpu 0,1,2,3
python train_classifier.py --config config/classifier/train_classifier_mini_kd.yaml --res_type resnet36_bottle --teacher_res_type resnet18_bottle --gpu 0,1,2,3
```
#### Training Meta-Baseline
* Training
```
python train_meta.py --config config/meta/train_meta_mini.yaml --res_type resnet12 --gpu 0,1
python train_meta.py --config config/meta/train_meta_mini.yaml --res_type resnet18 --gpu 0,1
python train_meta.py --config config/meta/train_meta_mini.yaml --res_type resnet36 --gpu 0,1
```
* Knowledge Distillation (classifier teacher)
```
python train_meta.py --config config/meta/train_meta_mini_kd.yaml --res_type resnet36_bottle --teacher_res_type resnet12_bottle --gpu 0,1
python train_meta.py --config config/meta/train_meta_mini_kd.yaml --res_type resnet36_bottle --teacher_res_type resnet18_bottle --gpu 0,1
```
* Knowledge Distillation (meta teacher)
```
python train_meta.py --config config/meta/train_meta_mini_kd.yaml --res_type resnet36_bottle --teacher_res_type resnet12_bottle  --teacher_meta_model --gpu 0,1
python train_meta.py --config config/meta/train_meta_mini_kd.yaml --res_type resnet36_bottle --teacher_res_type resnet18_bottle  --teacher_meta_model --gpu 0,1
```
**Note:** 
1. For the definitions of different distillation losses, you can find them in `Few-shot Learning/utils/__init__.py`.
