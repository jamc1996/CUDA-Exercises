#include <stdio.h>
#include <math.h>
#include <string.h>

#include "matreduce.h"

__global__ void GPU_row_reduce(float* Vgpu, float *aDatagpu, int n, int m)
{
	int i;
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < n)
	{
		for (i=0;i<m;i++){
			Vgpu[idx] += fabs(aDatagpu[(idx*m)+i]);
		}
	}
}

__global__ void GPU_col_reduce(float* Vgpu, float* aDatagpu, int n, int m)
{
	int i;
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < m)
	{
		for (i=0;i<n;i++){
			Vgpu[idx] += fabs(aDatagpu[idx+(i*n)]);
		}
	}
	
}


void CPU_row_reduce(float* V, float** A, int n, int m )
{
	int i,j;
	for (i = 0; i<n; i++)
	{
		for (j = 0; j<m; j++)
		{
			V[i] += fabs(A[i][j]);
		}
	}
}

void CPU_col_reduce(float* V, float** A, int n, int m)
{
	int i,j;
	for (i=0; i<n; i++)
	{
		for (j=0; j<m; j++)
		{
			V[j] += fabs(A[i][j]);
		}
	}
}

void CPU_sum_vec(float* tot, float* V, int n)
{
	int i;
	for (i=0; i<n; i++)
	{
		*tot += V[i];
	}
}





