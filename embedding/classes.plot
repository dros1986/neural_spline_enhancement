# set size 0.5, 0.75
set size 0.7, 0.75
set terminal pdf enhanced
set output "classes.pdf"
set termoption dash
set ylabel "{/Symbol D\}E_{76}"

set key off

set xtics nomirror rotate 90
set ytics 7,2,17
set xrange [-1.5:21]
set yrange [9:18]
set style fill solid
set boxwidth 0.5

DATAFILE = "classes-subject.txt"

set label "Subject" at 2.5, 16 center
set label "Illum." at 8.5, 16 center
set label "Location" at 13, 16 center
set label "Daytime" at 18, 16 center


plot "classes-subject.txt" using 0:3:xtic(1) with boxes fc rgb "#000080", \
     "classes-illumination.txt" using (7.5 + $0):3:xtic(1) with boxes fc rgb "#F07000", \
     "classes-location.txt" using (12 + $0):3:xtic(1) with boxes fc rgb "#000080", \
     "classes-time.txt" using (16.5 + $0):3:xtic(1) with boxes fc rgb "#F07000"