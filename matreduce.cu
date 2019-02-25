#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "matreduce.h"

__global__ void GPU_row_reduce(float* Vgpu, float *aDatagpu, int n, int m)
{
	int i;
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < n)
	{
		for (i=m;i--;){
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
		for (i=n;i--;){
			Vgpu[idx] += fabs(aDatagpu[idx+(i*n)]);
		}
	}
	
}

extern void run_tests(int n, int m, int seed)
{
	int i;
	clock_t begin, end;
	double time_CPUr, time_CPUc, time_GPUr, time_GPUc;

	float *aData, *Arowred, *Acolred;
	float** A;
	
	float *adgpu, *argpu, *acgpu;
	
	srand(seed);
	aData = (float*) malloc(sizeof(float)*n*m);
	A = (float**) malloc(sizeof(float*)*n);
	Arowred = (float*) malloc(sizeof(float)*n);
	Acolred = (float*) malloc(sizeof(float)*m);	


	cudaMalloc ((void**) &adgpu, sizeof(float)*n*m);
	cudaMalloc ((void**) &argpu, sizeof(float)*n);
	cudaMalloc ((void**) &acgpu, sizeof(float)*m);

	

	for ( i=0; i<n*m; i++)
	{
		aData[i] =( (float)(drand48()) *2.0 )-1.0;
	}
	for ( i=0; i<n; i++)
	{
		A[i] = &aData[i*m];
	}
	
	memset(Arowred,0.0,sizeof(float)*n);
	memset(Acolred,0.0,sizeof(float)*m);
		
	cudaMemcpy(adgpu,aData,sizeof(float)*n*m, cudaMemcpyHostToDevice);
	cudaMemcpy(argpu,Arowred,sizeof(float)*n, cudaMemcpyHostToDevice);
	cudaMemcpy(acgpu,Acolred,sizeof(float)*m, cudaMemcpyHostToDevice);

	int block_size = 8;
	dim3 dimBlock(block_size);
	dim3 dimGrid( (n/dimBlock.x) + (!((n)%dimBlock.x)?0:1) );

	begin = clock();

	CPU_row_reduce(Arowred,A,n,m);

	end = clock();
	
	time_CPUr = (double)(end - begin)/CLOCKS_PER_SEC;

	begin = clock();

	CPU_col_reduce(Acolred,A,n,m);

	end = clock();

	time_CPUc = (double)(end - begin)/CLOCKS_PER_SEC;	

	begin = clock();

	GPU_row_reduce<<<dimGrid,dimBlock>>>(argpu,adgpu,n,m);

	end = clock();
	
	time_GPUr = (double)(end - begin)/CLOCKS_PER_SEC;

	begin = clock();

	GPU_col_reduce<<<dimGrid,dimBlock>>>(acgpu,adgpu,n,m);

	end = clock();
	
	time_GPUc = (double)(end - begin)/CLOCKS_PER_SEC;

	printf("Time of CPU on row reduction: %lf\n",time_CPUr);
	printf("Time of CPU on column reduction: %lf\n",time_CPUc);
	printf("Time of GPU on row reduction: %lf\n",time_GPUr);
	printf("Time of GPU on column reduction: %lf\n",time_GPUc);


	free(Arowred);
	free(Acolred);
	free(aData);
	free(A);
	cudaFree(argpu);
	cudaFree(acgpu);
}


extern void CPU_row_reduce(float* V, float** A, int n, int m )
{
	int i,j;
	for (i = n; i--;)
	{
		for (j = n; j--;)
		{
			V[i] += fabs(A[i][j]);
		}
	}
}

extern void CPU_col_reduce(float* V, float** A, int n, int m)
{
	int i,j;
	for (i=n; i--;)
	{
		for (j=m; j--;)
		{
			V[j] += fabs(A[i][j]);
		}
	}
}

extern void CPU_sum_vec(float* tot, float* V, int n)
{
	int i;
	for (i=0; i<n; i++)
	{
		*tot += V[i];
	}
}





