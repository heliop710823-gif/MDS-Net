# [MDS-Net] Multi-axial Directional Strip Attention and Spatial-Semantic Aggregation Network for Drone-Satellite Geo-Localization

Code for MDS-Net.

## Prerequisites

- torch
- torchvision
- pyyaml
- tqdm
- numpy
- scipy
- matplotlib
- pillow
- timm==0.5.4

## Train and Test
We provide scripts to complete CCR training and testing
* Change the **data_dir** and **test_dir** paths in **run.sh** and then run:

For University-1652:
```shell
bash run_university.sh
```
For SUES-200:
```shell
bash run_sues.sh
```

## Dataset & Preparation
Download [University-1652](https://github.com/layumi/University1652-Baseline) upon request and put them under the `./data/` folder. You may use the request [template](https://github.com/layumi/University1652-Baseline/blob/master/Request.md).

Download [SUES-200](https://github.com/Reza-Zhu/SUES-200-Benchmark) upon request and put them under the `./data/` folder.

## Model weights
This is our weight download address. If you want to use it, please download and unzip it and put it in the root directory of the CCR project.

Address: [Here (Baidu Cloud Disk)](https://pan.baidu.com/s/1yE6c2gvY6Iv6riHDLtOxaw) Extraction code: 0000

## Reference
- **University-1652**: [pdf](https://arxiv.org/abs/2002.12186)|[code](https://github.com/layumi/University1652-Baseline)
- **SUES-200**: [pdf](https://arxiv.org/abs/2204.10704)|[code](https://github.com/Reza-Zhu/SUES-200-Benchmark)
