#!/bin/bash
echo "p, s, t" >> log.txt
for i in {1..16}
do
    for k in {1..5}
    do
    	echo -n "$i, " >> log.txt
    	mpirun -np $i ./int_calc 1000000 >> log.txt
    	tail -1 log.txt
    done
done
