# ARG1: title
# ARG2: data file
# ARG3: reference performance

set size 0.5, 0.75
set terminal pdf enhanced
set termoption dash
set logscale x
set xlabel "# images"
set ylabel "{/Symbol D\}E_{76}"
set title ARG1

set ytics 1
# set yrange [8:14]

reference=ARG3 + 0

plot ARG2 using 2:18:28 with filledcurve fc rgb "grey80" notitle, \
     ARG2 using 2:8 with linespoints notitle lc rgb "black" lw 2 pt 7 ps 0.6, \
     reference with lines notitle lc rgb "black" dt 2 lw 2
