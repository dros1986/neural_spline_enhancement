#!/bin/bash


MODEL="../models/rgb_srgb_np_10_nf_8_lr_0.000100_wd_0.100000_avgpool_abcde_emb_best.pth"


printf " "
for exp in A B C D E; do
    printf "\t%6s" $exp
done
printf "\n"

for user in F G H I J; do
    printf "%s" $user
    for exp in A B C D E; do
	RES=$(python3 evalprofile.py  $MODEL $user $exp | grep DE7 | cut -d' ' -f2)
	printf "\t%6.3f" $RES
    done
    printf "\n"
done
