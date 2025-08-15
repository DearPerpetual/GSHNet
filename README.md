## Introduction
Status: Archive (code is provided as-is, no updates expected)
### Inference code
Code for reproducing results in the paper __GSHNet: A Gated and Saliency-guided Hierarchical Network for Power Line Segmentation in aerial images__.

## Network Architecture
![pipeline](https://github.com/DearPerpetual/GSHNet/blob/main/figure/fig1.png)

## Results
<p align="center">
  <img src="https://github.com/DearPerpetual/GSHNet/blob/main/figure/105.png" width="200"/>
  <img src="https://github.com/DearPerpetual/GSHNet/blob/main/figure/116.png" width="200"/>
  <img src="https://github.com/DearPerpetual/GSHNet/blob/main/figure/144.png" width="200"/>
  <img src="https://github.com/DearPerpetual/GSHNet/blob/main/figure/186.png" width="200"/>
</p>

<p align="center">
  <img src="https://github.com/DearPerpetual/GSHNet/blob/main/figure/218.png" width="200"/>
  <img src="https://github.com/DearPerpetual/GSHNet/blob/main/figure/59.png" width="200"/>
  <img src="https://github.com/DearPerpetual/GSHNet/blob/main/figure/64.png" width="200"/>
  <img src="https://github.com/DearPerpetual/GSHNet/blob/main/figure/186.png" width="200"/>
</p>


## Require
Please `pip install` the following packages:
- torch
- torchvision
- opencv-python
- pillow
- timm
- numpy

## Development Environment

Running on Ubuntu 16.04 system with pytorch.

## Inference （Take the PLDU dataset as an example）
### step 1: Install python packages in requirement.txt.

### step 2: Download the weight `output/model/model.pth` to the root directory.

- Model weights and test results download link：[64ix](https://pan.baidu.com/s/1rFHj47XtQNIj9PRh3_YpVg).

### step 3: Run the following script to obtain Seg results in the testing image.
  `python run.py --test True`

__Note: The resolution of all test images is adjusted to `360x540`.__

