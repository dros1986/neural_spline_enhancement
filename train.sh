#!/bin/bash

DB=/mnt/data/dataset/fivek


# python3 main.py \
# 	-i $DB/raw \
# 	-e $DB/c \
# 	-bs 50 -np 10 -nf 8 \
# 	-tr $DB/train_mit.txt $DB/test_mit_random250.txt \
# 	-en 2018-06-14-baseline-de94


python3 main.py \
	-i $DB/raw \
	-e $DB/c \
	-bs 50 -np 10 -nf 8 \
	-tr $DB/train_mit.txt $DB/test_mit_random250.txt \
	-en 2018-06-19-resnet50dropout
