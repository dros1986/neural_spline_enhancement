set size 2.5/5.0, 2.5/3.0
set terminal pdf
set output "embedding.pdf"
set size ratio -1
set key box bottom right

set yrange [-0.9:0.6]
set xrange [-0.75:0.75]

c1 = 0.03112502
c2 = 0.00471976
u11 = -0.6850168947081396
u12 = -0.7285271813490678
u21 = -0.7285271813490678
u22 = 0.6850168947081396

plot "embedding.txt" using 2:3:1 every  ::0::4 with labels point pt 7 lc rgb "#808080" offset 1.5 notitle, \
     "embedding.txt" using 2:3:1 every  ::5::9 with labels point pt 7 lc rgb "#000080" offset 1.5 title "voting", \
     "embedding.txt" using 2:3:1 every  ::10::14 with labels point pt 7 lc rgb "#F07000" offset 1.5 title "fitting"


