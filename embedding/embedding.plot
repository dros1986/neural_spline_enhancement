set size 3.0/5.0, 3.0/3.0
set terminal pdf
set output "embedding.pdf"
set size ratio -1

set yrange [-0.5:0.5]
set xrange [-0.5:0.5]

c1 = 0.03112502
c2 = 0.00471976
u11 = -0.6850168947081396
u12 = -0.7285271813490678
u21 = -0.7285271813490678
u22 = 0.6850168947081396

plot c2 + (x - c1) * (u11 / u12) with lines lc rgb "gray" dt 2 lw 2 notitle, \
     c2 + (x - c1) * (u21 / u22) with lines lc rgb "gray" dt 2 lw 2 notitle, \
     "embedding.txt" using 2:3:1 with labels point pt 7 offset 1.5 notitle
     
#     "embedding.txt" using 4:5:1 with labels point pt 7 offset 1 notitle
