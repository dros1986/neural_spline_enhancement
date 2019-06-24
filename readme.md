## Evaluation

It is possible to replicate the results of the paper with the following command:

``` python
clear && python regen.py \
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
