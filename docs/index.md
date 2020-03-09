## Authors

- Simone Bianco ([simone.bianco@unimib.it](mailto:simone.bianco@disco.unimib.it)) - University of Milano-Bicocca<br>
- Claudio Cusano ([claudio.cusano@unipv.it](mailto:claudio.cusano@unipv.it)) - University of Pavia<br>
- Flavio Piccoli ([flavio.piccoli@unimib.it](mailto:flavio.piccoli@unimib.it)) - University of Milano-Bicocca<br>
- Raimondo Schettini ([raimondo.schettini@unimib.it](mailto:raimondo.schettini@unimib.it)) - University of Milano-Bicocca<br>

## Abstract
In this work we present SpliNet, a novel CNN-based method that estimates a global color transform for the enhancement of raw images. The method is designed to improve the perceived quality of the images by reproducing the ability of an expert in the field of photo editing.
The transformation applied to the input image is found by a convolutional neural network specifically trained for this purpose. More precisely, the network takes as input a raw
image and produces as output one set of control points for each of the three color channels. Then, the control points are interpolated with natural cubic splines and the resulting functions are globally applied to the values of the input pixels to produce the output image.
Experimental results compare favorably against recent methods in the state of the art on the MIT-Adobe FiveK dataset.
Furthermore, we also propose an extension of the SpliNet in which a single neural network is used to model the style of multiple reference retouchers by embedding them into a user space.  The style of new users can be reproduced without retraining the network, after a quick modeling stage in which they are positioned in the user space on the basis of their preferences on a very small set of retouched images.

## Paper

Please include the following reference in your paper if you mention the method (the paper is not available yet):

```
@inproceedings{personalizedimageenhancement2020,
  author = {Bianco, Simone and Cusano, Claudio and Piccoli, Flavio and Schettini, Raimondo},
  title = {Personalized Image Enhancement using Neural Spline Color Transform},
  booktitle = {IEEE Transactions on Image Processing},
  pages = {0--0},
  year = {2020}
}
```

## Overview

Given a RAW image, the method estimates the enhancement performed by the user. Two approaches are proposed. The first one learns to mimic the enhancement of the user through a new training of the system:

![pipeline single user](https://github.com/dros1986/neural_spline_enhancement/raw/master/docs/pipe_single.png)


The second approach is an extension of the first and allows to learn the user style just with few enhanced images and does not require training. An optimization mechanism assigns a signature to the new user that encodes its style and use it to adapt the pretrained system:

![pipeline adaptation](https://github.com/dros1986/neural_spline_enhancement/raw/master/docs/pipe_multi.png)


## Installing and running the software

The method is implemented in the python programming language and uses the pytorch framework for deep learning.
It has been tested on a workstation equiped with a single NVIDIA Titan-Xp GPU and with the Ubuntu 16.04 operating system,
python version 3.7.4, CUDA 10.1, CUDNN 7.0.5.

To install the software the following steps are suggested (others may work as well).

from a terminal:
```
git clone https://github.com/dros1986/neural_spline_enhancement.git
```

### Processing the images

The pretrained models of each expert of the FiveK dataset are already in the repository in the folder _models_.

The paths to the images to process must be placed in a text file (one path per line). Paths should start after the path specified in _input\_dir_. Images can be enhanced with the following command:

``` bash
python regen.py \
--input_dir <DATASET_DIR>/raw \
--test_list <DATASET_DIR>/test-list.txt \
--out_dir ./regen \
--out_dir_splines ./regen_splines \
--model ./models/expC.pth \
--batchsize 10 \
--npoints 10 \
--nfilters 8 \
--nexperts 1
```

Sobstitute <DATASET_DIR> with the directory of the dataset. If you want to test different experts, just change the model with _exp<LETTER>.pth_ of the corresponding expert where _<LETTER>_ can be _A,B,C,D,E_.


### Training a new model

Similarly to the testing procedure, training a new model requires
that the paths to the images are listed in a text file to form a training set.

The command to give is then:
``` bash
python train.py \
--input_dir <DATASET_DIR>/raw/ \
--experts_dir <DATASET_DIR>/expC \
--train_list <DATASET_DIR>/train-list.txt \
--val_list <DATASET_DIR>/test-list.txt \
--batchsize 10 \
--npoints 10 \
--nfilters 8 \
--nexperts 1
```
where `train-list.txt` is the file listing the training images and `input_dir` and `experts_dir` are respectively the folders containing the raw and the expert-enhanced images. The training script supports many options (type `python train.py -h` to list them).  Default values are those used in the paper.
