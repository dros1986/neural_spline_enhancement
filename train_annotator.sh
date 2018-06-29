#!/bin/bash


# # for WD in 0.001 0.1 1 10 100; do
# for WD in 0.3 0.03 0.003 0.0001 0.0003 0.00001 0.00003 0.000001 0.000003; do
#     python3 annotation.py --batch_size 50 --iterations 10000 --weight_decay $WD --model_file annotator-$WD.pth > annotator-$WD.log
# done


python3 annotation.py --batch_size 50 --iterations 7500 --weight_decay 0.03 --model_file annotator.pth > annotator.log
