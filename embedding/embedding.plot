set size 3.0/5.0, 3.0/3.0
set terminal pdf
set output "embedding.pdf"
set size ratio -1

set yrange [-0.5:0.5]
set xrange [-0.5:0.5]


plot "-" using 1:2:3 with labels point pt 7 offset 1 notitle
-0.3469 -0.1896 A 
0.1869 -0.3367 B
-0.1448 0.1571 C
0.1845 0.0535 D
0.2760 0.3392 E
e

