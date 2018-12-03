# Learning towards Minimum Hyperspherical Energy

By Weiyang Liu*, Rongmei Lin*, Zhen Liu*, Lixin Liu*, Zhiding Yu, Bo Dai, Le Song

### License
MHE and SphereFace+ are released under the MIT License (refer to the LICENSE file for details).

### Contents
0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Requirements](#requirements)
0. [Usage](#usage)
0. [Results](#results)
0. [SphereFace+ for face recognition](#sphereface-plus)
0. [Note](#note)


### Introduction

The repository contains the tensorflow implementation of Minimum Hyperspherical Energy (MHE) and Caffe implementation of SphereFace+. The code will be released shortly.


### Citation

If you find our work useful in your research, please consider to cite:

    @article{LiuNIPS18,
      title={Learning towards Minimum Hyperspherical Energy},
      author={Liu, Weiyang and Lin, Rongmei and Liu, Zhen and Liu, Lixin and Yu, Zhiding and Dai, Bo and Song, Le},
      journal={NIPS},
      year={2018}
      }

### Requirements
1. `Python 3.6` (`Python 2.7` needs to modify the unpickle funtion in train.py)
2. `TensorFlow`
3. `numpy`

### Usage

#### Part 1: Clone the repositary
  - Clone the repositary.

	```Shell
	git clone https://github.com/wy1iu/MHE.git
	```

#### Part 2: CIFAR-100
  - Train and test the model with the following code:

	```Shell
	cd $MHE_ROOT/code
	python train.py --model_name=mhe --power_s=0
	```
  - The parameters include `model_name: [baseline, mhe, half_mhe]` and `power_s: [0, 1, 2, a0, a1, a2]`. `mhe` denotes the full-space MHE regularization, `half_mhe` represents the half-space MHE, and `power_s` is the parameter s (see equation 1 in the paper) to control the formulation of MHE (`0,1,2` are the Euclidean distance, and `a0,a1,a2` use the angles).
  - For different datasets, you may need adjuest to hyperparameter for the entire MHE regularizaiton.
### Results

### sphereface-plus

SphereFace+ is implemented in Caffe and therefore is independently host [here](https://github.com/wy1iu/sphereface-plus).

### Note


