#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

#include "matreduce.h"

extern void run_time_tests(int n, int m, int seed);

int main(int argc, char* argv[])
{
  int n = 10000;
  int m = 10000;
  int seed = 123456;
  int timeflag = 0;
  int c=0;
  while ((c=getopt(argc,argv,"tn:m:s:")) != -1)
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
    case '?':
      return 1;
    }
  }
 if (timeflag == 1)
  {
   run_time_tests(n,m,seed);     
  } 

  return 0;
}
