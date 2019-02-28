#Following based on Makefile to makefileExternC from 1st batch of cuda source code.

# Compilers and commands
CC = gcc
NVCC = nvcc


objects = main.o matreduce.o

main: $(objects)
	$(NVCC) -o main $(objects)

#main.o: main.cu
#	$(NVCC) -c $<

matreduce.o: matreduce.cu matreduce.h
	$(NVCC) -c $<

test: main
	./main -t -n 10000 -m 10000

.PHONY: clean
clean:
	rm -f $(objects) main
