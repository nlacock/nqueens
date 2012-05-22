#!/bin/bash

MAX_ITERS=1000

make
l="0"
bs="512"
while [ $bs -gt 8 ]
do
    l=$bs
    while [ $l -gt 2 ]
    do

	echo "BOARDSIZE: $bs LOCALSIZE $l"
	#srun -N1 --gres=gpu:1 ./ir_sequential $i $MAX_ITERS 
	/software/slurm/bin/srun -N1 --gres=gpu:1 ./ir_parallel $bs $l $MAX_ITERS | grep -v Q
	echo -e "\n\n"
	l=$[$l/2]
    done
bs=$[$bs/2]
done
