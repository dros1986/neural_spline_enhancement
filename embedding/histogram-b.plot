set size 0.4, 0.6
set terminal pdf enhanced
set output "histogram-b.pdf"
# set xlabel "b"
set ylabel "P(b)"
set title "b"

# set key outside right
set key off

set xtics -120, 60, 120 nomirror
set format y "10^{%L}"
set logscale y
set ytics 1e-7, 100, 1e-1

DATAFILE = "histogram-b.txt"

plot DATAFILE using 1:2 with lines title "Expert C"  lc rgb "#000080" lw 2, \
     DATAFILE using 1:3 with lines title "Processed"  lc rgb "#F07000" lw 2
