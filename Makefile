# note, this Makefile works for cuda.acad.cis.udel.edu
#  no guarantees that it works for any other opencl system
CXX = g++
CXX_FLAGS = -I/software/cuda-sdk/OpenCL/common/inc -lrt -lOpenCL
SRUNX = /software/slurm/bin/srun -N1 --gres=gpu:1
CC = gcc


ALL = ir_parallel ir_parallel_2 ir_sequential

ir_parallel: ir_parallel.cpp
	$(CXX) -o $@ $^ $(CXX_FLAGS)

ir_parallel_2: ir_parallel_2.cpp
	$(CXX) -o $@ $^ $(CXX_FLAGS)


ir_sequential: ir_sequential.cpp
	$(CXX) -o $@ $^ $(CXX_FLAGS)

run:
	srun -N1 --gres=gpu:1 ./ir_parallel

clean:
	rm -f $(ALL) *~
