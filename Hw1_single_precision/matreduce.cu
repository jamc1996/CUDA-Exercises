#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "matreduce.h"


__global__ void dummy_calc(int n)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int i,k=0;
	if (idx < n)
	{
		for (i=200;i--;)
		{
			k+=i;	
		}
	}
}

__global__ void GPU_row_reduce(float* Vgpu, float *aDatagpu, int n, int m)
/* Function to sum rows of matrix in parallel on the GPU.*/
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
/* Function to sum the columns of a matrix in parallel on the GPU. */
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
/* Function to sequentially sum a vector on GPU. */
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
/* Function to allow parallel summing of a vector.
Function must be called repeatedly from CPU as otherwise synchronization doesn't work.*/
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;	
	if (idx < i)
	{
		Vgpu[idx] += Vgpu[i + odd + idx];
	}
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
	*tot = 0;
	//using for(i=n; i--;) was slower here I think it confused the cache.	
	for (i=0; i<n; i++)
	{
		*tot += V[i];
	}
}



extern void run_basic_tests(int n, int m, int seed, int block_size)
/* Function to check that all the functions are working. */
{
	// Variables declared for use on CPU.
	int i;
	float *aData, *Arowred, *Acolred;
	float** A;
	float total_rc, total_cc, total_rg, total_cg;	
	float error1,error2,error3,error4;	

	// Variables for GPU
	float *adgpu, *argpu, *acgpu, *gpu_rsum, *gpu_csum;

	// Memory allocated and values filled into CPU variables.
	srand48(seed);
	aData = (float*) malloc(sizeof(float)*n*m);
	A = (float**) malloc(sizeof(float*)*n);
	Arowred = (float*) malloc(sizeof(float)*n);
	Acolred = (float*) malloc(sizeof(float)*m);	
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

	

	// Layout of GPU set:		
	dim3 dimBlock(block_size);
	dim3 dimGrid( (n/dimBlock.x) + (!((n)%dimBlock.x)?0:1) );

	// CPU calculations done and results printed to screen.
	CPU_row_reduce(Arowred,A,n,m);
	CPU_reduce_vec(&total_rc, Arowred, n);

	printf("=============================================================\n");
	printf("\t\tCPU (Row Reduction) \n\n");
	printf("\n\t\tFully Reduced Matrix: %f\n",total_rc);
	printf("=============================================================\n");

	CPU_col_reduce(Acolred,A,n,m);
	CPU_reduce_vec(&total_cc, Acolred, m);
	
	printf("=============================================================\n");
	printf("\t\tCPU (Column Reduction) \n\n");
	printf("\n\t\tFully Reduced Matrix: %f\n",total_cc);
	printf("=============================================================\n");


	// Extra GPU memory allocated and filled.	
	cudaMalloc ((void**) &adgpu, sizeof(float)*n*m);
	cudaMalloc ((void**) &gpu_rsum, sizeof(float));
	cudaMalloc ((void**) &argpu, sizeof(float)*n);
	cudaMalloc ((void**) &gpu_csum, sizeof(float));
	cudaMalloc ((void**) &acgpu, sizeof(float)*m);
	cudaMemcpy(adgpu,aData,sizeof(float)*n*m, cudaMemcpyHostToDevice);

	// GPU calculations done and results printed to screen.	
	GPU_row_reduce<<<dimGrid,dimBlock>>>(argpu,adgpu,n,m);
	GPU_sequential_sum<<<dimGrid,dimBlock>>>(argpu,n,gpu_rsum);
	cudaMemcpy(&total_rg,gpu_rsum,sizeof(float), cudaMemcpyDeviceToHost);

	printf("=============================================================\n");
	printf("\t\tGPU Performance (Row Reduction) (with Sequential Vector Reduction) \n\n");
	printf("\n\t\tFully Reduced Matrix: %f\n",total_rg);
	printf("=============================================================\n");
	
	GPU_col_reduce<<<dimGrid,dimBlock>>>(acgpu,adgpu,n,m);
	i = m;
	int odd;
	while (i>2){
		odd = i%2;		
		i/=2;
		GPU_parallel_sum_vec<<<dimGrid,dimBlock>>>(acgpu, i,odd);
		i += odd;
	}
	GPU_parallel_sum_vec<<<dimGrid,dimBlock>>>(acgpu,1,0);
	cudaMemcpy(&total_cg,acgpu,sizeof(float), cudaMemcpyDeviceToHost);

	printf("=============================================================\n");
	printf("\t\tGPU Performance (Column Reduction) (with Parallel Vector Reduction)\n\n");
	printf("\n\t\tFully Reduced Matrix: %f\n",total_cg);
	printf("=============================================================\n");

	// Error between different methods of Matrix Reduction calculated and results printed:
	error1 = fabs(((total_rc - total_cc)/total_rc)*100.0);
	error2 = fabs(((total_rg - total_cg)/total_rc)*100.0);
	error3 = fabs(((total_rc - total_rg)/total_rc)*100.0);
	error4 = fabs(((total_cc - total_cg)/total_rc)*100.0);
	
	printf("=============================================================\n");
	printf("\t\tError between Methods:\n");
	printf("CPU Percentage Error Between Summing by Row vs by Column: %.5f%%\n",error1);
	printf("GPU Percentage Error Between Summing by Row vs by Column: %.5f%%\n",error2);
	printf("CPU vs GPU Percentage Error (by Row): %.5f%%\n",error3);
	printf("CPU vs GPU Percentage Error (by Column): %.5f%%\n",error4);
	printf("=============================================================\n");
	
	// Memory freed:
	free(Arowred);
	free(Acolred);
	free(aData);
	free(A);
	cudaFree(argpu);
	cudaFree(acgpu);
	cudaFree(adgpu);
	cudaFree(gpu_rsum);
	cudaFree(gpu_csum);
}


extern void run_time_tests(int n, int m, int seed, int block_size)
/* Function to test the speed of functions running on the GPU. */
{
	//Variables for use on CPU
	int i;
	float *aData, *Arowred, *Acolred;
	float** A;
	float total;	

	//Variables for use of GPU
	float *adgpu, *argpu, *acgpu, *rsum_gpu, *csum_gpu;

	// Variables for timing execution of functions	
	clock_t begin, end, mem_time, calc_time;
	double time_allocating, time_calculating, time_copying, total_time;
	
	// CPU allocation and assignment
	srand48(seed);
	aData = (float*) malloc(sizeof(float)*n*m);
	A = (float**) malloc(sizeof(float*)*n);
	Arowred = (float*) malloc(sizeof(float)*n);
	Acolred = (float*) malloc(sizeof(float)*m);	
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
		
	// GPU Grid Layout calculated.
	dim3 dimBlock(block_size);
	dim3 dimGrid( (n/dimBlock.x) + (!((n)%dimBlock.x)?0:1) );

	// Timing of CPU row reduce
	begin = clock();
	CPU_row_reduce(Arowred,A,n,m);
	end = clock();
	total_time = (double)(end - begin)/CLOCKS_PER_SEC;
	
	// Results printed to Screen
	printf("=============================================================\n");
	printf("\t\tCPU Performance (Row Reduction) \n\n");
	printf("\n\t\tTotal time spent: %lf\n",total_time);
	printf("=============================================================\n");

	// Timing of CPU column reduce	
	begin = clock();
	CPU_col_reduce(Acolred,A,n,m);
	end = clock();
	total_time = (double)(end - begin)/CLOCKS_PER_SEC;

	// Results printed to Screen
	printf("=============================================================\n");
	printf("\t\tCPU Performance (Column Reduction) \n\n");
	printf("\n\t\tTotal time spent: %lf\n",total_time);	
	printf("=============================================================\n");

	// Dummy Function run to try and remove any warm up time.
	dummy_calc<<<dimGrid,dimBlock>>>(n);

	// Timing of GPU row reduce	
	begin = clock();

	cudaMalloc ((void**) &adgpu, sizeof(float)*n*m);
	cudaMalloc ((void**) &argpu, sizeof(float)*n);
	cudaMemcpy(adgpu,aData,sizeof(float)*n*m, cudaMemcpyHostToDevice);

	mem_time = clock();

	GPU_row_reduce<<<dimGrid,dimBlock>>>(argpu,adgpu,n,m);

	calc_time = clock();

	cudaMemcpy(Arowred,argpu,sizeof(float)*n, cudaMemcpyDeviceToHost);

	end = clock();

	// Timings calculated
	time_allocating = (double)(mem_time - begin)/CLOCKS_PER_SEC;
	time_calculating = (double)(calc_time - mem_time)/CLOCKS_PER_SEC;
	time_copying = (double)(end - calc_time)/CLOCKS_PER_SEC;
	total_time = time_allocating + time_calculating + time_copying;

	// Results printed to screen
	printf("=============================================================\n");
	printf("\t\tGPU Performance (Row Reduction) \n\n");
	printf("\tTime spent allocating memory space: %lfs\n",time_allocating);
	printf("\tTime spent in calculation: %lfs\n", time_calculating);
	printf("\tTime spent copying back to Host: %lfs\n", time_copying);
	printf("\n\t\tTotal time spent: %lf\n",total_time);
	printf("=============================================================\n");

	
	
	// adgpu refreed and dummy function run again so conditions as similar as possible.
	cudaFree(adgpu);
	dummy_calc<<<dimGrid,dimBlock>>>(n);

	// Timing of GPU Column Reduction
	begin = clock();

	cudaMalloc((void**) &adgpu, sizeof(float)*n*m);
	cudaMalloc ((void**) &acgpu, sizeof(float)*m);
	cudaMemcpy(adgpu,aData,sizeof(float)*n*m, cudaMemcpyHostToDevice);

	mem_time = clock();

	GPU_col_reduce<<<dimGrid,dimBlock>>>(acgpu,adgpu,n,m);

	calc_time = clock();

	cudaMemcpy(Acolred,acgpu,sizeof(float)*m, cudaMemcpyDeviceToHost);

	end = clock();

	// Times calculated
	time_allocating = (double)(mem_time - begin)/CLOCKS_PER_SEC;
	time_calculating = (double)(calc_time - mem_time)/CLOCKS_PER_SEC;
	time_copying = (double)(end - calc_time)/CLOCKS_PER_SEC;
	total_time = time_allocating + time_calculating + time_copying;

	// Results printed
	printf("=============================================================\n");
	printf("\t\tGPU Performance (Column Reduction) \n\n");
	printf("\tTime spent allocating memory space: %lfs\n",time_allocating);
	printf("\tTime spent in calculation: %lfs\n", time_calculating);
	printf("\tTime spent copying back to Host: %lfs\n", time_copying);
	printf("\n\t\tTotal time spent: %lf\n",total_time);
	printf("=============================================================\n");

	// Timing of CPU vector reduction:
	begin = clock();

	total = 0;	
	CPU_reduce_vec(&total, Arowred, n);

	end = clock();

	total_time = (double)(end - begin)/CLOCKS_PER_SEC;

	// Results printed to Screen	
	printf("=============================================================\n");
	printf("\t\tCPU Vector Reduction \n\n");
	printf("\tReducing the vector of summed rows on CPU: %lf\n",total);
	printf("\n\t\tTotal time spent: %lf\n",total_time);
	printf("=============================================================\n");
	

	// Timing of GPU sequential vector reduction.
	begin = clock();

	cudaMalloc ((void**) &rsum_gpu, sizeof(float));
	GPU_sequential_sum<<<dimGrid,dimBlock>>>(argpu,n,rsum_gpu);
	cudaMemcpy(&total,rsum_gpu,sizeof(float), cudaMemcpyDeviceToHost);

	end = clock();

	total_time = (double)(end - begin)/CLOCKS_PER_SEC;

	// Results printed to screen
	printf("=============================================================\n");
	printf("\t\tGPU Sequential Vector Reduction \n\n");
	printf("\tReducing the vector of summed rows: %lf\n",total);
	printf("\n\t\tTotal time spent: %lf\n",total_time);
	printf("=============================================================\n");

	// Timing of GPU parallel vector reduction:
	begin = clock();
	
	cudaMalloc ((void**) &csum_gpu, sizeof(float));
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

	// Results printed to screen:	
	printf("=============================================================\n");
	printf("\t\tGPU Parallel Vector Reduction \n\n");
	printf("\tReducing the vector of summed rows: %lf\n",total);
	printf("\n\t\tTotal time spent: %lf\n",total_time);
	printf("=============================================================\n");

	// Memory freed
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

extern void write_times(int n, int m, int seed, int block_size)
/* Function to write different GPU speeds to file. It does not do this particularly cleanly,
but printed them in such a way that I could get a Python function  */
{
	FILE *fp;
	fp = fopen("results.txt","a");
	fprintf(fp,"The following is mostly raw data for use in graphing.\n");
	fprintf(fp,"The columns represent:\nTime on CPU Reducing Matrix and Result by row and column, Times for various steps of GPU reduction and results.");
	fprintf(fp,"\nResults for %d by %d matrix, with block size %d\n",n,m,block_size);
	//Variables for use on CPU
	int i;
	float *aData, *Arowred, *Acolred;
	float** A;
	float total;	

	//Variables for use of GPU
	float *adgpu, *argpu, *acgpu;

	// Variables for timing execution of functions	
	clock_t begin, end, mem_time, calc_time;
	double time_allocating, time_calculating, time_copying, total_time;
	
	// CPU allocation and assignment
	srand48(seed);
	aData = (float*) malloc(sizeof(float)*n*m);
	A = (float**) malloc(sizeof(float*)*n);
	Arowred = (float*) malloc(sizeof(float)*n);
	Acolred = (float*) malloc(sizeof(float)*m);	
	for ( i=0; i<n*m; i++)
	{
		aData[i] =( (float)(drand48()) *2.0 )-1.0;
	}
	for ( i=0; i<n; i++)
	{
		A[i] = &aData[i*m];
	}
		
	// GPU Grid Layout calculated.
	dim3 dimBlock(block_size);
	dim3 dimGrid( (n/dimBlock.x) + (!((n)%dimBlock.x)?0:1) );


	// Timing of CPU row reduce
	begin = clock();
	CPU_row_reduce(Arowred,A,n,m);
	end = clock();
	total_time = (double)(end - begin)/CLOCKS_PER_SEC;

	// Results printed to file
	float tote;
	begin = clock();
	CPU_reduce_vec(&tote,Arowred,n);
	end = clock();
	total_time += (double)(end - begin)/CLOCKS_PER_SEC;


	// Results printed to file
	fprintf(fp, "%lf, %f, ", total_time, tote);

	// Timing of CPU column reduce	
	begin = clock();
	CPU_col_reduce(Acolred,A,n,m);
	end = clock();
	total_time = (double)(end - begin)/CLOCKS_PER_SEC;

	begin = clock();
	CPU_reduce_vec(&tote,Acolred,m);
	end = clock();
	total_time += (double)(end - begin)/CLOCKS_PER_SEC;

	// Results printed to file
	fprintf(fp, "%lf, %f, ",total_time, tote);

	// Dummy Function run to try and remove any warm up time.
	dummy_calc<<<dimGrid,dimBlock>>>(n);
	float* rtotal;
	
	// Timing of GPU row reduce	
	begin = clock();

	cudaMalloc ((void**) &rtotal, sizeof(float));
	cudaMalloc ((void**) &adgpu, sizeof(float)*n*m);
	cudaMalloc ((void**) &argpu, sizeof(float)*n);
	cudaMemcpy(adgpu,aData,sizeof(float)*n*m, cudaMemcpyHostToDevice);


	mem_time = clock();

	GPU_row_reduce<<<dimGrid,dimBlock>>>(argpu,adgpu,n,m);
	GPU_sequential_sum<<<dimGrid,dimBlock>>>(argpu,n,rtotal);

	calc_time = clock();

	cudaMemcpy(&total,rtotal,sizeof(float), cudaMemcpyDeviceToHost);

	end = clock();

	// Timings calculated
	time_allocating = (double)(mem_time - begin)/CLOCKS_PER_SEC;
	time_calculating = (double)(calc_time - mem_time)/CLOCKS_PER_SEC;
	time_copying = (double)(end - calc_time)/CLOCKS_PER_SEC;
	total_time = time_allocating + time_calculating + time_copying;

	// Results printed to file.
	fprintf(fp,"%lf,  %lf,  %lf, %lf,%f ",time_allocating,time_calculating, time_copying, total_time, total);

	
	
	// adgpu refreed and dummy function run again so conditions as similar as possible.
	cudaFree(adgpu);
	dummy_calc<<<dimGrid,dimBlock>>>(n);
	float* ctotal;
	// Timing of GPU Column Reduction
	begin = clock();

	cudaMalloc((void**) &ctotal, sizeof(float));
	cudaMalloc((void**) &adgpu, sizeof(float)*n*m);
	cudaMalloc ((void**) &acgpu, sizeof(float)*m);
	cudaMemcpy(adgpu,aData,sizeof(float)*n*m, cudaMemcpyHostToDevice);

	mem_time = clock();

	GPU_col_reduce<<<dimGrid,dimBlock>>>(acgpu,adgpu,n,m);
	GPU_sequential_sum<<<dimGrid,dimBlock>>>(acgpu,n,ctotal);

	calc_time = clock();	

	cudaMemcpy(&total,ctotal,sizeof(float), cudaMemcpyDeviceToHost);

	end = clock();

	// Times calculated
	time_allocating = (double)(mem_time - begin)/CLOCKS_PER_SEC;
	time_calculating = (double)(calc_time - mem_time)/CLOCKS_PER_SEC;
	time_copying = (double)(end - calc_time)/CLOCKS_PER_SEC;
	total_time = time_allocating + time_calculating + time_copying;

	// Results printed to file.
	fprintf(fp,"%lf,  %lf,  %lf, %lf, %f \n",time_allocating,time_calculating, time_copying, total_time, total);

	fclose(fp);
	// Memory freed
	free(Arowred);
	free(Acolred);
	free(aData);
	free(A);
	cudaFree(argpu);
	cudaFree(acgpu);
	cudaFree(adgpu);
}




