#!/bin/bash


for F in 1 2 3 4; do
    cut -d, -f $(( $F + 1 )) annotations.csv | sort | uniq > labels-$F.txt
    cat labels-$F.txt | while read L; do
	cut -d, -f 1,$(( $F + 1 )) annotations.csv | grep "$L" | join -t, test-list.txt - | cut -d, -f 1 > test-"$F"-"$L".txt
    done
done
