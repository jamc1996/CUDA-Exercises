#include "cudaset.h"
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

/*		cudaset.cu -- functions for gpu modelling of heat propagation in a radiator.
 * 	
 *		Author: John Cormican
 *	
 *		Purpouse: To model heat flow quickly using the GPU.
 *	
 *		Usage: Use run_cuda_radiator to call the host functions.
 */

#define BLOCK_SIZE 128
#define REDUCE_BLOCK_SIZE 128

surface<void, 2> array_surf;
surface<void, 2> reduced_surf;

__global__ void transformSurfaceToGlobal ( double* adgpu, size_t pitch, int dim_x, int dim_y) 
/* Function to transform surface memory to a global array (pitched). */
{
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        int idy = blockIdx.y*blockDim.y + threadIdx.y;
        
        if ( (idx < dim_x) && (idy < dim_y) ) {
		double *row = (double*)((char*)adgpu + idy*pitch);
                // Read from input surface, save into the Global memory
                surf2Dread(&(row[idx]), array_surf, idx*sizeof(double) , idy); 
        }
}

__global__ void  transformReducedSurface ( double* redgpu, int dim_x)
/* Function to transform first entries of a 2d surface to a 1-d global array. */
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	if(idy==0 && idx<dim_x)
	{
		surf2Dread(&redgpu[idx], reduced_surf, 0, idx);
	}

}


__device__ void prop_in_block(double *a, double *b, int thread_id)
/* Device function to update one a values of one of shared memory arrays. */
{
        a[thread_id] = ((1.9*b[thread_id-2]) + (1.5*(b[thread_id-1]))  + b[thread_id] + (0.5*b[(thread_id+1)]) + (0.1*b[(thread_id+2)]) )  ;
        a[thread_id] /= (double)5.0;
}


__global__ void GPU_ROW_REDUCE(int n, int m, int bsize, int k, int first)
/* Kernel function to find the average of each row. */
{
	__shared__ double a_array[REDUCE_BLOCK_SIZE];

	int idx = blockIdx.x*blockDim.x + threadIdx.x;
        int idy = blockIdx.y*blockDim.y;
        int thread_id = threadIdx.x;
        int block_space = blockIdx.x*blockDim.x;
        int block_id = blockIdx.x;

	if (block_space < k && idy < n)
        {
		if ( k < block_space + REDUCE_BLOCK_SIZE)
		{
		
			k = k% REDUCE_BLOCK_SIZE;
			
		}
		else
		{
			k = bsize;
		}
                if (thread_id < k)
                {
			if (first == 1)
                        {
                                surf2Dread(&a_array[thread_id], array_surf, idx*sizeof(double), idy);
				a_array[thread_id]/=(double)(m);
			}
                        else
                        {
	                        surf2Dread(&a_array[thread_id], reduced_surf, idx*sizeof(double), idy);
			}
                }
		int i, mod;
		while (k>2)
                {
                	mod = k%2;
                        i = (k/2);
                        if (thread_id<i)
                        {
                                a_array[thread_id] += a_array[thread_id+i+mod];
                        }
                        k = i+mod;
                        __syncthreads();
                }
		if (thread_id == 0)
		{
			a_array[0] += a_array[1];
			surf2Dwrite(a_array[thread_id], reduced_surf, block_id*sizeof(double), idy);			
		}
	}
}

__global__ void GPU_PROPAGATE(int n, int m, int early_stop)
/* Host function to simulate 7 propagations of heat flow through radiator. */
{
  	// Shared memory each thread needs to access.
	__shared__ double a_array[BLOCK_SIZE];
  	__shared__ double b_array[BLOCK_SIZE];  
  	__shared__ int unedited[2];

	// Identifiers for threads	
	int idy = blockIdx.y*blockDim.y;
	int thread_id = threadIdx.x;
	
	// Variables for accessing correct part of global memory.
	int i;
  	int return_area = (blockIdx.x)*4;
	int section_head = (m + ((return_area - 14)%m))%m;
  	int low_lim = 1, up_lim = 30;

	// Warning flags not to edit first two rows: 
	if (thread_id == 0)
	{
		unedited[0] = -1;
		unedited[1] = -1;
	}

	__syncthreads();		

	// All blocks called together so syncthreads usable inside this if statement
	if (return_area < m && idy<n)
 	{
		int xindex = (section_head+thread_id)%m;
  		surf2Dread(&a_array[thread_id], array_surf, xindex*sizeof(double), idy); 	 	
		// Update warning flags:
		if (xindex == 0 || xindex == 1)
    		{
      			unedited[xindex] = thread_id;
			b_array[thread_id] = a_array[thread_id];
    		}
		__syncthreads();
		// Update steps:
  		for (i=0; i< early_stop; i++)
  		{
    			if (thread_id>low_lim && thread_id<up_lim)
    			{	
      				if (i%2 == 0)
   	   			{
					if (unedited[0] != thread_id && unedited[1] != thread_id)
        				{
        					prop_in_block(b_array,a_array,thread_id);
					}
      				}
      				if (i%2 == 1)
      				{
					if (unedited[0] != thread_id && unedited[1] != thread_id)
					{
						prop_in_block(a_array,b_array,thread_id);
					}
      				}
    			}
    			low_lim += 2;
    			up_lim -= 2;
    			__syncthreads();
  		}
	
	// Write back to global memory:	
  		if (thread_id < 18 && thread_id > 13)
  		{
			surf2Dwrite(b_array[thread_id], array_surf, (return_area + thread_id -14)*sizeof(double),idy);
  		}
  	}
}

void run_cuda_radiator(Matrix* A, double* av_heats, int p, int t_flag)
/* CPU function which calls necessary kernels to simulate cylindrical radiator.  */
{
  int i;
  double *adgpu;
  double *redgpu;
  int N = A->nRows*A->nColumns; 
  int reduce_bsize = 128;
  int block_size = 128;
  dim3 dimBlock(block_size,1);
  dim3 dimGrid((A->nColumns/4) + (!(A->nColumns%dimBlock.x)?0:1),A->nRows);
  size_t ag_size;

  cudaEvent_t start, finish;
  float time_alloc, time_trans1, time_comp,time_av, time_trans2;
  cudaEventCreate(&start);
  cudaEventCreate(&finish);

  cudaEventRecord(start,0);
  
  cudaMallocPitch((void**)&adgpu,&ag_size, (size_t)(sizeof(double)*A->nColumns), A->nRows);
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float2>();
 // cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(sizeof(double)*8,0,0,0,cudaChannelFormatKindFloat);
  cudaArray* cuSurfArray;
  cudaArray* cuReducedArray;
  cudaMallocArray(&cuSurfArray, &channelDesc, A->nColumns, A->nRows, cudaArraySurfaceLoadStore);
  cudaMallocArray(&cuReducedArray, &channelDesc, (A->nColumns/reduce_bsize)+1, A->nRows, cudaArraySurfaceLoadStore);
  
  cudaEventRecord(finish,0);
  cudaEventSynchronize(start);
  cudaEventSynchronize(finish);
  cudaEventElapsedTime(&time_alloc, start, finish);
  
  cudaEventRecord(start, 0);
  cudaMemcpyToArray(cuSurfArray, 0, 0, A->data_, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaBindSurfaceToArray(array_surf, cuSurfArray);
  cudaBindSurfaceToArray(reduced_surf, cuReducedArray);

  cudaEventRecord(finish, 0);

  cudaEventSynchronize(start);
  cudaEventSynchronize(finish);

  cudaEventElapsedTime(&time_trans1, start, finish);

  cudaEventRecord(start, 0);

  // All calculation done in batches of 7 propagations 
  int batch_size = (block_size/4)-1;
  for (i=0;i<(p/batch_size);i++)
  {
    GPU_PROPAGATE<<<dimGrid,dimBlock>>>(A->nRows,A->nColumns,batch_size);
  }
  GPU_PROPAGATE<<<dimGrid,dimBlock>>>(A->nRows,A->nColumns,p%batch_size);
  
  cudaEventRecord(finish, 0);   

  cudaEventSynchronize(start);
  cudaEventSynchronize(finish);

  cudaEventElapsedTime(&time_comp, start, finish);


  cudaEventRecord(start, 0);
  dim3 dimBlock2(reduce_bsize,1);
  dim3 dimGrid2((A->nColumns/reduce_bsize) + (!(A->nColumns%dimBlock.x)?0:1),A->nRows);
  int m = A->nColumns;
  int temp;
  int first_flag = 1;
  while(m>1)
  {
    temp = 0;
    if ( m%reduce_bsize > 0)
    {
      temp=1;
    }
    GPU_ROW_REDUCE <<<dimGrid2, dimBlock2 >>> (A->nRows, A->nColumns,reduce_bsize,m,first_flag);
    m/=reduce_bsize;
    m+=temp;    
    first_flag = 0; 
  }
  cudaEventRecord(finish, 0);

  cudaEventSynchronize(start);
  cudaEventSynchronize(finish);

  cudaEventElapsedTime(&time_av, start, finish);

  cudaEventRecord(start, 0);
  cudaMalloc((void**) &redgpu, sizeof(double)*A->nRows);

  transformSurfaceToGlobal <<<dimGrid, dimBlock>>> (adgpu, ag_size, A->nColumns, A->nRows);
  transformReducedSurface <<<dimGrid, dimBlock>>> (redgpu, A->nRows);

  cudaMemcpy(av_heats,redgpu,sizeof(double)*A->nRows,cudaMemcpyDeviceToHost);
  cudaMemcpy2D(A->data_, A->nColumns*sizeof(double), adgpu, ag_size, A->nColumns*sizeof(double), A->nRows, cudaMemcpyDeviceToHost);

  cudaEventRecord(finish, 0);

  cudaEventSynchronize(start);
  cudaEventSynchronize(finish);

  cudaEventElapsedTime(&time_trans2, start, finish);

  if (t_flag == 1)
  { 
    printf("Time spent allocating = %f micro-seconds.\n",time_alloc*1000);  
    printf("Time spent transferring to GPU = %f micro-seconds.\n",time_trans1*1000);  
    printf("Time spent computing matrix = %f micro-seconds.\n",time_comp*1000);  
    printf("Time spent computing averages = %f micro-seconds.\n",time_av*1000);  
    printf("Time spent transferring from GPU = %f micro-seconds.\n",time_trans2*1000);  
    printf("Time spent total = %f\n", 1000*(time_alloc+time_trans1+time_comp+time_av+time_trans2));
  }

  cudaFree(adgpu);
  cudaFree(redgpu);
  cudaFreeArray(cuSurfArray);
  cudaFreeArray(cuReducedArray);
}















