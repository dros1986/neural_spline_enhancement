#!/bin/bash

for S in vote fit; do
    LINE=4
    for e in a b c d e; do
	DATA="$S-profile-results-$e.txt"
	SUMMARY="$S-profile-summary-$e.txt"
	PDF="$S-profile-$e.pdf"
	CROP=${PDF%.pdf}-crop.pdf

	# REF is the performance obtained learning from the expert
	REF=$(head -$LINE noprofiles.txt | tail -1 | cut -d' ' -f 1)
	LINE=$(( $LINE + 1 ))

	# Creaty a summary from the raw data
	grep AVG -A2 $DATA | tr $'\n' ' ' | sed "s/--/\n/g" > $SUMMARY

	# Make the plot
	gnuplot -c profile.plot "Expert ${e^^}" $SUMMARY $REF > $PDF

	# Crop the result
	pdfcrop $PDF
	mv $CROP $PDF
    done
done



gnuplot embedding.plot

pdfcrop embedding.pdf
mv embedding-crop.pdf embedding.pdf
