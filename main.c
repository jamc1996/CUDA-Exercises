#include "matreduce.h"

extern void run_tests(int n, int m, int seed);

int main()
{
  run_tests(10000,10000,123456);     

  return 0;
}
