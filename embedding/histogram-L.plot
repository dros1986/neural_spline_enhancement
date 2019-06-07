set size 0.4, 0.6
set terminal pdf enhanced
set output "histogram-L.pdf"
# set xlabel "L"
set ylabel "P(L)"
set title "L"

set key off

# set xtics 5
set xtics nomirror
set format y "10^{%L}"
set logscale y

# set ytics 2
# set yrange [5:11.1]

DATAFILE = "histogram-L.txt"

plot DATAFILE using 1:2 with lines notitle  lc rgb "#000080" lw 2, \
     DATAFILE using 1:3 with lines notitle  lc rgb "#F07000" lw 2
