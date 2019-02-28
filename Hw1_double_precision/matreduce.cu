#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "matreduce.h"

__global__ void GPU_row_reduce(float* Vgpu, float *aDatagpu, int n, int m)
{
	int i;
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < n)
	{
		Vgpu[idx] =0.0;
		for (i=m;i--;)
		{
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
		Vgpu[idx] = 0.0;
		for (i=n;i--;)
		{
			Vgpu[idx] += fabs(aDatagpu[idx+(i*n)]);
		}
	}
	
}
__global__ void GPU_sequential_sum(float* Vgpu, int n, float *sum)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;	
	int i;
	if (idx == 0)
	{
		for(i=n;i--;)
		{
			*sum += Vgpu[i];
		}
	}
}

__global__ void GPU_parallel_sum_vec(float* Vgpu, int i, int odd)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;	
//	i /=  2;
	if (idx < i)
	{
		Vgpu[idx] += Vgpu[i + odd + idx];
	}
}

extern void run_time_tests(int n, int m, int seed, int block_size)
{
	int i;
	clock_t begin, end, mem_time, calc_time;
	double time_allocating, time_calculating, time_copying, total_time;
	float *aData, *Arowred, *Acolred;
	float** A;
	float total;	
	float *adgpu, *argpu, *acgpu, *rsum_gpu, *csum_gpu;
	
	srand(seed);
	aData = (float*) malloc(sizeof(float)*n*m);
	
	A = (float**) malloc(sizeof(float*)*n);
	Arowred = (float*) malloc(sizeof(float)*n);
	Acolred = (float*) malloc(sizeof(float)*m);	

	cudaMalloc ((void**) &rsum_gpu, sizeof(float));
	cudaMalloc ((void**) &csum_gpu, sizeof(float));
	cudaMalloc ((void**) &adgpu, sizeof(float)*n*m);
	

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
		
	//cudaMemcpy(acgpu,Acolred,sizeof(float)*m, cudaMemcpyHostToDevice);

	dim3 dimBlock(block_size);
	dim3 dimGrid( (n/dimBlock.x) + (!((n)%dimBlock.x)?0:1) );

	begin = clock();

	CPU_row_reduce(Arowred,A,n,m);

	end = clock();
	
	total_time = (double)(end - begin)/CLOCKS_PER_SEC;

	printf("=============================================================\n");
	printf("\t\tCPU Performance (Row Reduction) \n\n");

	printf("\n\t\tTotal time spent: %lf\n",total_time);
	
	printf("=============================================================\n");
	


	begin = clock();

	CPU_col_reduce(Acolred,A,n,m);

	end = clock();
	
	total_time = (double)(end - begin)/CLOCKS_PER_SEC;

	printf("=============================================================\n");
	printf("\t\tCPU Performance (Column Reduction) \n\n");

	printf("\n\t\tTotal time spent: %lf\n",total_time);
	
	
	printf("=============================================================\n");
	
	begin = clock();
	
	cudaMalloc ((void**) &argpu, sizeof(float)*n);
	cudaMemcpy(adgpu,aData,sizeof(float)*n*m, cudaMemcpyHostToDevice);

	mem_time = clock();

	GPU_row_reduce<<<dimGrid,dimBlock>>>(argpu,adgpu,n,m);

	calc_time = clock();

	cudaMemcpy(Arowred,argpu,sizeof(float)*n, cudaMemcpyDeviceToHost);

	end = clock();

	printf("=============================================================\n");
	printf("\t\tGPU Performance (Row Reduction) \n\n");

	time_allocating = (double)(mem_time - begin)/CLOCKS_PER_SEC;
	time_calculating = (double)(calc_time - mem_time)/CLOCKS_PER_SEC;
	time_copying = (double)(end - calc_time)/CLOCKS_PER_SEC;
	total_time = time_allocating + time_calculating + time_copying;

	printf("\tTime spent allocating memory space: %lfs\n",time_allocating);
	printf("\tTime spent in calculation: %lfs\n", time_calculating);
	printf("\tTime spent copying back to Host: %lfs\n", time_copying);

	printf("\n\t\tTotal time spent: %lf\n",total_time);

	printf("=============================================================\n");

	cudaFree(adgpu);
	cudaMalloc((void**) &adgpu, sizeof(float)*n*m);

	begin = clock();
	
	cudaMalloc ((void**) &acgpu, sizeof(float)*m);
	cudaMemcpy(adgpu,aData,sizeof(float)*n*m, cudaMemcpyHostToDevice);

	mem_time = clock();

	GPU_col_reduce<<<dimGrid,dimBlock>>>(acgpu,adgpu,n,m);

	calc_time = clock();

	cudaMemcpy(Acolred,acgpu,sizeof(float)*m, cudaMemcpyDeviceToHost);

	end = clock();

	printf("=============================================================\n");
	printf("\t\tGPU Performance (Column Reduction) \n\n");

	time_allocating = (double)(mem_time - begin)/CLOCKS_PER_SEC;
	time_calculating = (double)(calc_time - mem_time)/CLOCKS_PER_SEC;
	time_copying = (double)(end - calc_time)/CLOCKS_PER_SEC;
	total_time = time_allocating + time_calculating + time_copying;

	printf("\tTime spent allocating memory space: %lfs\n",time_allocating);
	printf("\tTime spent in calculation: %lfs\n", time_calculating);
	printf("\tTime spent copying back to Host: %lfs\n", time_copying);

	printf("\n\t\tTotal time spent: %lf\n",total_time);
	printf("=============================================================\n");
	


	printf("=============================================================\n");
	printf("\t\tGPU Sequential Vector Reduction \n\n");

	begin = clock();
	
	GPU_sequential_sum<<<dimGrid,dimBlock>>>(argpu,n,rsum_gpu);
	
	cudaMemcpy(&total,rsum_gpu,sizeof(float), cudaMemcpyDeviceToHost);

	end = clock();

	total_time = (double)(end - begin)/CLOCKS_PER_SEC;

	printf("\tReducing the vector of summed rows: %lf\n",total);
	
	printf("\n\t\tTotal time spent: %lf\n",total_time);
	printf("=============================================================\n");
	
	
	printf("=============================================================\n");
	printf("\t\tGPU Parallel Vector Reduction \n\n");

	begin = clock();
	
	i = n;
	int odd;
	while (i>2){
		odd = i%2;		
		i/=2;
		GPU_parallel_sum_vec<<<dimGrid,dimBlock>>>(argpu, i,odd);
		i += odd;
	}
	GPU_parallel_sum_vec<<<dimGrid,dimBlock>>>(argpu,1,0);
	
	cudaMemcpy(&total,argpu,sizeof(float), cudaMemcpyDeviceToHost);

	end = clock();

	total_time = (double)(end - begin)/CLOCKS_PER_SEC;

	printf("\tReducing the vector of summed rows: %lf\n",total);
	
	printf("\n\t\tTotal time spent: %lf\n",total_time);
	printf("=============================================================\n");

	
	printf("=============================================================\n");
	printf("\t\tCPU Vector Reduction \n\n");

	begin = clock();

	total = 0;	
	CPU_reduce_vec(&total, Arowred, n);

	end = clock();

	total_time = (double)(end - begin)/CLOCKS_PER_SEC;

	printf("\tReducing the vector of summed rows on CPU: %lf\n",total);
	
	printf("\n\t\tTotal time spent: %lf\n",total_time);
	printf("=============================================================\n");
	


	free(Arowred);
	free(Acolred);
	free(aData);
	free(A);
	cudaFree(argpu);
	cudaFree(acgpu);
	cudaFree(adgpu);
	cudaFree(rsum_gpu);
	cudaFree(csum_gpu);
}


extern void CPU_row_reduce(float* V, float** A, int n, int m )
/* Function to sequentially reduce the rows of A to a vector
of the sums of the absolute values of their entries*/
{
	int i,j;
	//using for(i=n; i--;) was slower here I think it confused the cache.
	for (i = 0; i<n; i++ )
	{
		for (j = 0; j<m; j++)
		{
			V[i] += fabs(A[i][j]);
		}
	}
}

extern void CPU_col_reduce(float* V, float** A, int n, int m)
/* Function to sequentially reduce the columns of A to a vector
of the sums of the absolute values of their entries*/
{
	int i,j;
	//using for(i=n; i--;) was slower here I think it confused the cache.
	// I got a decent speed up by summing different entries of V on each iteration.
	for ( i = 0; i<n; i++)
	{
		for ( j=0; j<m; j++)
		{
			V[j] += fabs(A[i][j]);
		}
	}
}

extern void CPU_reduce_vec(float* tot, float* V, int n)
/* Function to sequentially reduce a vector to the sum of its values. */
{
	int i;
	//using for(i=n; i--;) was slower here I think it confused the cache.	
	for (i=0; i<n; i++)
	{
		*tot += V[i];
	}
}





