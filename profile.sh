#!/bin/bash

DB=/mnt/ssd/dataset/fivek

python3 -m cProfile main.py \
	-i $DB/raw \
	-e $DB/c \
	-bs 50 -np 10 -nf 32 \
	-tr $DB/train_mit.txt $DB/test_mit_random250.txt \
	-ne 10 \
	-en profile
