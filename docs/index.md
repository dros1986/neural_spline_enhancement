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
