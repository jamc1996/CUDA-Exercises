#include <stdio.h>
#include <stdlib.h>

#include "matreduce.h"

int main(){
	int n = 20;
	int m = 20;
	int i,j;
	float* aData;
	float** A;
	float *aDgpu, *Vgpu;
	float *V;

	aData = (float*) malloc(sizeof(float)*n*m);	
	A = (float**) malloc(sizeof(float*)*n);
	V = (float*) malloc(sizeof(float)*n);
	

	cudaMalloc ((void**) &aDgpu, sizeof(float)*n*m);
	cudaMalloc ((void**) &Vgpu, sizeof(float)*n);


	for(i=0;i<n*m;i++) 
	{
		aData[i] =(float) (drand48()*2.0)-1.0;
	}
	for(i=0;i<n;i++)
	{
		A[i] = &aData[i*m];
	}
	printf("Printing Matrix:\n");
	for (i=0;i<n;i++){
		for(j=0;j<m;j++){
			printf("%lf ",A[i][j]);
		}
		printf("\n");
	}

	memset(V,0.0,sizeof(float)*n);

	cudaMemcpy(aDgpu, aData, sizeof(float)*n*m, cudaMemcpyHostToDevice);
	cudaMemcpy(Vgpu, V, sizeof(float)*n, cudaMemcpyHostToDevice);

	int block_size = 8;
	dim3 dimBlock(block_size);
	dim3 dimGrid( (n/dimBlock.x) + (!((n)%dimBlock.x)?0:1) );

	GPU_row_reduce<<<dimGrid,dimBlock>>>(Vgpu,aDgpu,n,m);

	cudaMemcpy(V,Vgpu,sizeof(float)*n, cudaMemcpyDeviceToHost);

	float* V2;
	V2 = (float*) malloc(sizeof(float)*n);
	CPU_row_reduce(V2,A,n,m);

	
	printf("Summed by row (GPU):\n");
	for(i=0; i<m;i++){
		printf("%f\n",V[i]);

	}
	float tot;
	CPU_sum_vec(&tot,V2,n);	

	printf("Summed by row (CPU):\n");
	printf("%f\n",tot);

	free(V);
	free(V2);
	free(aData);
	free(A);
	cudaFree(Vgpu);
	cudaFree(aDgpu);
}
