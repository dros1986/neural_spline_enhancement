#!/bin/bash

# DB=/mnt/data/dataset/fivek
DB=/mnt/ssd/dataset/fivek
GT=/mnt/ssd/dataset/fivek/c
# GT=/mnt/ssd/dataset/fivek/experts_prophoto_8bit/256x256/c

# python3 main.py \
# 	-i $DB/raw \
# 	-e $GT \
# 	-bs 50 -np 10 -nf 8 \
# 	-tr $DB/train_mit.txt $DB/test_mit_random250.txt \
# 	-en baseline


python3 main.py \
	-i $DB/raw \
	-e $GT \
	-bs 50 -np 10 -nf 8 \
	-tr $DB/train_mit.txt $DB/test_mit_random250.txt \
	-en color_transform_srgb


# python3 main.py \
# 	-i $DB/raw \
# 	-e $GT \
# 	-bs 50 -np 10 -nf 8 \
# 	-tr $DB/train_mit.txt $DB/test_mit_random250.txt \
# 	-en baseline_prophoto_wrong_de76
