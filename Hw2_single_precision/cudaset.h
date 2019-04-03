#ifndef CUDASET_H
#define CUDASET_H

#include "matrix.h"

#ifdef __cplusplus
extern "C"
{
  void run_cuda_radiator(Matrix* A, float* av_heats, int p, int t_flag);
}

#endif



#endif
