# set size 0.5, 0.75
set size 0.6, 0.75
set terminal pdf enhanced
set output "spline-nodes.pdf"
set termoption dash
set xlabel "Number of nodes"
set ylabel "Error"
# set ylabel "{/Symbol D\}E_{76}"

set key outside right

set xtics 5
set xtics nomirror

# set ytics 2
set yrange [5:11.1]

DATAFILE = "spline-nodes.txt"

plot DATAFILE using 1:2 with linespoints title "{/Symbol D\}E_{76}" lc rgb "#000080" lw 2 pt 7 ps 0.6, \
     DATAFILE using 1:3 with linespoints title "{/Symbol D\}E_{94}" lc rgb "#F07000" lw 2 pt 13 ps 0.6, \
     DATAFILE using 1:4 with linespoints title "{/Symbol D\}L" lc rgb "#707070" lw 2 pt 5 ps 0.6
