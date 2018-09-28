# set size 0.5, 0.75
set size 0.6, 0.75
set terminal pdf enhanced
set output "times.pdf"
set xlabel "Image size"
set ylabel "Processing time (ms)"

set key off

# set xtics 5
set logscale xy
set xtics nomirror rotate by 330 # 45 right
set xtics ("32{/Symbol \264}32" 32, "64{/Symbol \264}64" 64, "128{/Symbol \264}128" 128, "256{/Symbol \264}256" 256, "512{/Symbol \264}512" 512, "1024{/Symbol \264}1024" 1024, "2048{/Symbol \264}2048" 2048, "4096{/Symbol \264}4096" 4096)

# set ytics 2
# set yrange [5:11.1]

DATAFILE = "times.txt"

plot DATAFILE using 1:(1000 * $2) with linespoints notitle lc rgb "#000080" lw 2 pt 7 ps 0.6
