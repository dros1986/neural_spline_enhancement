Implementation of the paper:

**Personalized Image Enhancement using Neural Spline Color Transform**<br>
Transactions on Graphics, ACM<br>
[S. Bianco](http://www.ivl.disco.unimib.it/people/simone-bianco/ "Simone Bianco"), [C. Cusano](http://www.ivl.disco.unimib.it/people/claudio-cusano/ "Claudio Cusano"), [F. Piccoli](http://www.ivl.disco.unimib.it/people/flavio-piccoli/ "Flavio Piccoli"), [R. Schettini](http://www.ivl.disco.unimib.it/people/raimondo-schettini/ "Raimondo Schettini")

![Pipeline](https://github.com/dros1986/neural_spline_enhancement/blob/master/res/pipe1.png?raw=true)

## Inference

It is possible to replicate the results of the paper with the following command:

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
