#!/bin/bash

DB=/mnt/ssd/dataset/fivek

TEST_FILE=$DB/test_mit_random250.txt


if [ "$1." == "train." ]; then
    TEST_FILE=$DB/train_mit.txt
fi

echo $TEST_FILE

python3 main.py \
	-i $DB/raw \
	-e $DB/c \
	-bs 50 -np 10 -nf 32 -tr 0 -ts \
	$TEST_FILE \
	saved-models/spline_npoints_10_nfilters_32_2018-06-10-down4g_best_10.6878.pth \
	./output


#	spline_npoints_10_nfilters_32_2018-06-04-downsampling.pth \
