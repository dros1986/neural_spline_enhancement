set size 0.5, 0.75
set terminal pdf enhanced
set output "histogram-keys.pdf"
set termoption dash

set key Left reverse

set yrange [-10:10]
set xrange [-10:10]

set noborder
set noxtics
set noytics
set notitle
set noxlabel
set noylabel

plot 20 with lines title "Expert C" lc rgb "#000080" lw 2, \
     21 with lines title "Output" lc rgb "#F07000" lw 2
