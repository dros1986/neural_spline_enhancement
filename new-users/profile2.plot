set size 0.5, 0.75
set terminal pdf enhanced
set termoption dash
set logscale x
set xlabel "Number of images"
set ylabel "{/Symbol D\}E_{76}"
set title ARG1 offset 7, -2.7

set key off

set xtics nomirror (1,3,10,30)

set ytics 2
set xrange [1:30]
set yrange [10:23]

set multiplot

plot ARG2 using 2:18:28 with filledcurve fc rgb "#C0000080" notitle, \
     ARG3 using 2:18:28 with filledcurve fc rgb "#C0F07000" notitle, \
     ARG2 using 2:8 with linespoints title "Voting" lc rgb "#000080" lw 2 pt 7 ps 0.6, \
     ARG3 using 2:8 with linespoints title "Fitting" lc rgb "#F07000" lw 2 pt 13 ps 0.6
set datafile separator "," 
plot ARG4 using 1:3 with lines title "CF (one expert)" lc rgb "black" dt 2 lw 2, \
     ARG5 using 1:3 with lines title "CF (five experts)" lc rgb "red" dt 2 lw 2
