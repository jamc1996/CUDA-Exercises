#Following based on Makefile to makefileExternC from 1st batch of cuda source code.

# Compilers and commands
CC = gcc
NVCC = nvcc
LINK = nvcc

#Flags
CFLAGS = -W -Wall
NVCCFLAGS = -g -G --use_fast_math

objects = main.o matreduce.o

all: main

main: $(objects)
	$(NVCC) -o main $(objects)

matreduce.o: matreduce.cu matreduce.h
	$(NVCC) -c $<

test: main
	./main -t -n 10000 -m 10000

memcheck: main
	cuda-memcheck ./main

.PHONY: clean
clean:
	rm -f $(objects) main results.txt
