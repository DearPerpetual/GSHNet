## Introduction
Status: Archive (code is provided as-is, no updates expected)
### Inference code
Code for reproducing results in the paper __GSHNet: A Gated and Saliency-guided Hierarchical Network for Power Line Segmentation in aerial images__.

## Network Architecture
![pipeline](https://github.com/DearPerpetual/GSHNet/blob/main/figure/fig1.png)

## Test Results
<p align="center">
  <img src="https://github.com/DearPerpetual/GSHNet/blob/main/figure/105.png" width="200"/>
  <img src="https://github.com/DearPerpetual/GSHNet/blob/main/figure/116.png" width="200"/>
  <img src="https://github.com/DearPerpetual/GSHNet/blob/main/figure/144.png" width="200"/>
  <img src="https://github.com/DearPerpetual/GSHNet/blob/main/figure/186.png" width="200"/>
</p>

<p align="center">
  <img src="https://github.com/DearPerpetual/GSHNet/blob/main/figure/218.png" width="200"/>
  <img src="https://github.com/DearPerpetual/GSHNet/blob/main/figure/59.png" width="200"/>
  <img src="https://github.com/DearPerpetual/GSHNet/blob/main/figure/446.png" width="200"/>
  <img src="https://github.com/DearPerpetual/GSHNet/blob/main/figure/376.png" width="200"/>
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

Running on Ubuntu 16.04 system with pytorch，GPU 3060.

## Inference （Take the PLDU dataset as an example）
### step 1: Install python packages in requirement.txt.
### step 2: Download the PLDU dataset and store it in a suitable location.
### step 3: Download the weight `output/model/model.pth` to the root directory.
- Model weights download link：[kzv3](https://pan.baidu.com/s/1MidlYCwuZ-28I8FVnwf2JQ).
### step 4: Run the following script to obtain Seg results in the testing image.
  `python run.py --test True`
### step 5: You will get the test results in the output folder.
__Note: The resolution of all test images is adjusted to `360x540`.__
