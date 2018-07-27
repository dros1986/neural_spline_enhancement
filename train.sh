for de in 76; do
    python3 train.py \
	    -i /mnt/data/dataset/fivek/siggraph2018/256x256/raw \
	    -e /mnt/data/dataset/fivek/siggraph2018/256x256 \
	    -tl /mnt/data/dataset/fivek/siggraph2018/train1+2-list.txt \
	    -vl /mnt/data/dataset/fivek/siggraph2018/test-list.txt \
	    -nf 8 \
	    -ds avgpool \
	    -bs 30 \
	    -ne 100 \
	    -wd 0.1 \
	    -cs srgb \
	    -at rgb \
	    -de $de \
	    -en serial-$de
done

