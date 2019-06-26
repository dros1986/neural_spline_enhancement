#!/bin/bash


# Both strategies on the same plot
LINE=4
for e in a b c d e; do
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
    gnuplot -c profile2.plot "Expert ${e^^}" $SUMMARY1 $SUMMARY2 $REF > $PDF

    # Crop the result
    pdfcrop $PDF
    mv $CROP $PDF
done


# One plot per strategy
# for S in vote fit; do
#     LINE=4
#     for e in a b c d e; do
# 	DATA="$S-profile-results-$e.txt"
# 	SUMMARY="$S-profile-summary-$e.txt"
# 	PDF="$S-profile-$e.pdf"
# 	CROP=${PDF%.pdf}-crop.pdf

# 	# REF is the performance obtained learning from the expert
# 	REF=$(head -$LINE noprofiles.txt | tail -1 | cut -d' ' -f 1)
# 	LINE=$(( $LINE + 1 ))

# 	# Creaty a summary from the raw data
# 	grep AVG -A2 $DATA | tr $'\n' ' ' | sed "s/--/\n/g" > $SUMMARY

# 	# Make the plot
# 	gnuplot -c profile.plot "Expert ${e^^}" $SUMMARY $REF > $PDF

# 	# Crop the result
# 	pdfcrop $PDF
# 	mv $CROP $PDF
#     done
# done


gnuplot profile2-keys.plot
pdfcrop profile-keys.pdf
mv profile-keys-crop.pdf profile-keys.pdf

for P in embedding spline-nodes times histogram-L histogram-a histogram-b histogram-keys classes users; do
    gnuplot $P.plot
    pdfcrop $P.pdf
    mv $P-crop.pdf $P.pdf
done


for e in a b c d e; do
    DATA1="multi-$e-classes-subject.txt"
    DATA2="multi-$e-classes-illumination.txt"
    DATA3="multi-$e-classes-location.txt"
    DATA4="multi-$e-classes-time.txt"
    PDF="multi-$e-classes.pdf"
    CROP=${PDF%.pdf}-crop.pdf

    # Make the plot
    gnuplot -c multi-classes.plot "Expert ${e^^}" $DATA1 $DATA2 $DATA3 $DATA4 > $PDF

    # Crop the result
    pdfcrop $PDF
    mv $CROP $PDF
done


# gnuplot embedding.plot
# pdfcrop embedding.pdf
# mv embedding-crop.pdf embedding.pdf

# gnuplot profile2-keys.plot
# pdfcrop profile-keys.pdf
# mv profile-keys-crop.pdf profile-keys.pdf
