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
