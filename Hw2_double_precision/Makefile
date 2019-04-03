#Following based on Makefile to makefileExternC from 1st batch of cuda source code.

# Compilers and commands
CC = gcc
NVCC = nvcc
LINK = nvcc
DEL_FILE = rm -f

#Flags
CFLAGS = -W -Wall
NVCCFLAGS = --use_fast_math -arch=sm_30

objects = main.o setmatrix.o propagate.o cudaset.o

all: radiator

radiator: $(objects) matrix.h
	$(NVCC) $(NVCCFLAGS) -o radiator $(objects)

setmatrix.o: setmatrix.c setmatrix.h matrix.h
	$(CC) -c $<

propagate.o: propagate.c propagate.h matrix.h
	$(CC) -c $<

cudaset.o: cudaset.cu cudaset.h matrix.h
	$(NVCC) $(NVCCFLAGS) -c $<

#matreduce.o: matreduce.cu matreduce.h
#	$(NVCC) $(NVCCFLAGS) -c $<

test: radiator
	./radiator

memcheck: main
	cuda-memcheck ./radiator

.PHONY: clean
clean:
	$(DEL_FILE) $(objects) radiator
