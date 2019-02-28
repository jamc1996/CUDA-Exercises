#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

#include "matreduce.h"

extern void write_times(int n, int m, int seed, int block_size);
extern void run_time_tests(int n, int m, int seed, int block_size);
extern void run_basic_tests(int n, int m, int seed, int block_size);

int main(int argc, char* argv[])
{
  int n = 100;
  int m = 100;
  int seed = 123456;
  int timeflag = 0, fileflag = 0;
  int c=0;
  int block_size = 8;
  while ((c=getopt(argc,argv,"tfb:n:m:s:")) != -1)
  {
    switch(c)
    {
    case 'n':
      n = atoi(optarg);
      break;
    case 'm':
      m = atoi(optarg);
      break;
    case 's':
      seed = atoi(optarg);
      break;
    case 't':
      timeflag = 1;
      break;
    case 'f':
      fileflag = 1;
      break;
    case 'b':
      block_size = atoi(optarg); 
      break;
    case '?':
      return 1;
    }
  }

  if (timeflag == 1)
  {
   run_time_tests(n,m,seed,block_size);     
  }
  else if (fileflag == 1)
  {
    write_times(n, m, seed, block_size);
  }
  else 
  {
    run_basic_tests(n,m,seed,block_size);
  }

  return 0;
}
