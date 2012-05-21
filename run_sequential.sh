#!/bin/bash

MAX_ITERS=50000

make

for i in 10 20 30 40 50 60 70 80 90 100; do
	echo "BOARDSIZE: $i"
	#srun -N1 --gres=gpu:1 ./ir_sequential $i $MAX_ITERS 
	srun -N1 --gres=gpu:1 ./ir_sequential $i $MAX_ITERS | grep -v Q
	echo -e "\n\n"
done
