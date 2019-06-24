Implementation of the paper: <NONE>

![Pipeline](https://github.com/dros1986/neural_spline_enhancement/blob/master/res/pipe1.png?raw=true)

## Inference

It is possible to replicate the results of the paper with the following command:

``` bash
python regen.py \
--input_dir <DATASET_DIR>/raw \
--test_list <DATASET_DIR>/test-list.txt \
--out_dir ./regen \
--out_dir_splines ./regen_splines \
--model ./models/expertC.pth \
--batchsize 10 \
--npoints 10 \
--nfilters 8 \
--nexperts 1
```
