# set size 0.5, 0.75
set size 0.7, 0.75
set terminal pdf enhanced
set output "users.pdf"
set termoption dash
set ylabel "Votes (%)"

set key top left

set xtics nomirror rotate by 30 right # offset (0, 0.5)
set ytics nomirror 10
# set ytics 7,2,17
# set xrange [-1.5:21]
set yrange [0:42]
set style fill solid
set boxwidth 0.25

TOT1 = 707
TOT2 = 574

plot "users.txt" using 0:(100 * $2 / TOT1):xtic(1) with boxes fc rgb "#000080" title "FiveK", \
     "users-raise100.txt" using (0.25 + $0):(100 * $2 / TOT2) with boxes fc rgb "#F07000" title "Raise"
