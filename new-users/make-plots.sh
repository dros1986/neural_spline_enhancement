#!/bin/bash


# Both strategies on the same plot
LINE=4
for e in f g h i j; do
    DATA1="vote-profile-results-$e.txt"
    DATA2="fit-profile-results-$e.txt"
    SUMMARY1="vote-profile-summary-$e.txt"
    SUMMARY2="fit-profile-summary-$e.txt"
    PDF="profile-$e.pdf"
    CROP=${PDF%.pdf}-crop.pdf

    # REF is the performance obtained learning from the expert
    REF=$(head -$LINE noprofiles.txt | tail -1 | cut -d' ' -f 1)
    LINE=$(( $LINE + 1 ))

    # Creaty summaries from the raw data
    grep AVG -A2 $DATA1 | tr $'\n' ' ' | sed "s/--/\n/g" > $SUMMARY1
    grep AVG -A2 $DATA2 | tr $'\n' ' ' | sed "s/--/\n/g" > $SUMMARY2

    # Make the plot
    gnuplot -c profile2.plot "User ${e^^}" $SUMMARY1 $SUMMARY2 $REF > $PDF

    # Crop the result
    pdfcrop $PDF
    mv $CROP $PDF
done

gnuplot profile2-keys.plot
pdfcrop profile-keys.pdf
mv profile-keys-crop.pdf profile-keys.pdf

for P in embedding; do # spline-nodes times histogram-L histogram-a histogram-b histogram-keys classes users; do
    gnuplot $P.plot
    pdfcrop $P.pdf
    mv $P-crop.pdf $P.pdf
done

