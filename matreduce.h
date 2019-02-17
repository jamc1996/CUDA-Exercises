#ifndef MATREDUCE_H
#define MATREDUCE_H

__global__ void GPU_col_reduce(float* Vgpu, float* aDatagpu, int n, int m);
__global__ void GPU_row_reduce(float* Vgpu, float* aDatagpu, int n, int m);
void CPU_row_reduce(float* V, float** A, int n, int m);
void CPU_col_reduce(float* V, float** A, int n, int m);
void CPU_sum_vec(float* tot, float* V, int n);


#endif
