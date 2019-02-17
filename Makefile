

CC = gcc
NVCC = nvcc
objects = main.o matreduce.o

reduce_m: $(objects)
	$(NVCC) -o reduce_m $(objects)

main.o: main.cu
	$(NVCC) -c $<

matreduce.o: matreduce.cu matreduce.h
	$(NVCC) -c $<
