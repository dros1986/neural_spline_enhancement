set size 0.5, 0.75
set terminal pdf enhanced
set output "profile-keys.pdf"
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

plot 20 with linespoints title "Voting" lc rgb "#000080" lw 2 pt 7 ps 0.6, \
     21 with linespoints title "Fitting" lc rgb "#F07000" lw 2 pt 13 ps 0.6, \
     22 with lines title "Collab.Filt (best expert)" lc rgb "black" dt 2 lw 2, \
     22 with lines title "Collab.Filt (all experts)" lc rgb "red" dt 2 lw 2
