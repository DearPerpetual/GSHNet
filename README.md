## Introduction
Status: Archive (code is provided as-is, no updates expected)
### Inference code
Code for reproducing results in the paper __GSHNet: A Gated and Saliency-guided Hierarchical Network for Power Line Segmentation in aerial images__.

## Network Architecture
![pipeline](https://github.com/DearPerpetual/GSHNet/blob/main/figure/fig1.png)

## Results
<p align="center">
<img src="https://github.com/DearPerpetual/MFPLNet/blob/main/work_dirs/out/swin_t_tusimple/20240925_121139_lr_1e-03_b_8/visualization/clips_0530_00000_2.jpg", width="360">
<p align="center">
<img src="https://github.com/DearPerpetual/MFPLNet/blob/main/work_dirs/out/swin_t_tusimple/20240925_121139_lr_1e-03_b_8/visualization/clips_0530_00000_795.jpg", width="360">
</p>
<p align="center">
<img src="https://github.com/DearPerpetual/MFPLNet/blob/main/work_dirs/out/swin_t_tusimple/20240925_121139_lr_1e-03_b_8/visualization/clips_0530_00000_857.jpg", width="360">
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

