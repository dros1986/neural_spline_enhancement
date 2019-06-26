#1!/bin/bash

MODEL=../models/rgb_srgb_np_10_nf_8_lr_0.000100_wd_0.100000_avgpool_abcde_emb_best.pth


function evalm() {
    FILE=test-annotations/test-"$1"-"$2".txt
    X=$(python3 evalprofile.py $MODEL $3 $4 "$FILE" | grep DE76 | cut -d' ' -f 2)
    printf "\"%s\" %d %.2f\n" "$2" $(wc -l "$FILE" | cut -d' ' -f1) $X 
}


PROFILE="1 0 0 0 0"
for EXPERT in a b c d e; do
    echo $EXPERT "$PROFILE"
    for C in "abstract" "animal(s)" "man-made object" "nature" "person(s)" "unknown"; do
	evalm 1 "$C" $EXPERT "$PROFILE"
    done > multi-$EXPERT-classes-subject.txt

    for C in "artificial" "mixed" "sun or sky"; do
	evalm 2 "$C" $EXPERT "$PROFILE"
    done > multi-$EXPERT-classes-illumination.txt

    for C in "indoors" "outdoors" "unknown"; do
	evalm 3 "$C" $EXPERT "$PROFILE"
    done > multi-$EXPERT-classes-location.txt

    for C in "dawn or dusk" "day" "night" "unknown"; do
	evalm 4 "$C" $EXPERT "$PROFILE"
    done > multi-$EXPERT-classes-time.txt

    P1=$(echo $PROFILE | cut -d' ' -f1-4)
    PROFILE="0 $P1"
done
