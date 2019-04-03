#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>

#include "matrix.h"
#include "setmatrix.h"
#include "propagate.h"
#include "cudaset.h"

extern void run_cuda_radiator(Matrix *A,double* av_heats,int p,int t_flag);

/*	main.c -- program to run CPU and GPU modelling of a radiator
 *
 *	Author: John Cormican
 *
 *	Purpouse: To test results of CPU vs. GPU calculations wrt time/accuracy
 *
 *	Usage: Run with command line arguments, calls CPU functions from 
 *	propagate.c and GPU functions from cudaset.cu
 */




int main(int argc, char* argv[])
/* function to run and test other functions from other programs. */
{
  int n = 32, m=32, p=10, t_flag = 0; // default values for matrix dimensions/iterations
  int c,i,j;
  int show_av=0;
  double* av_heats;
  while ((c = getopt (argc, argv, "n:m:p:at")) != -1)
  {
    switch (c)
    {
    case 'n':
      n = atoi(optarg);
      break;
    case 'm':
      m = atoi(optarg);
      break;
    case 'p':
      p = atoi(optarg);
      break;
    case 'a':
      show_av = 1;
      break;
    case 't':
      t_flag = 1;
      break;
    case '?':
      return 1;
    }
  }
  Matrix A1,A2;
  initiate_matrix(&A2,n,m);

  struct timeval start, end;
  long int time;
  if(show_av == 1)
  {
    initiate_matrix(&A1,n,m);
    gettimeofday(&start,NULL);
    iterative_solver(&A1,p);
    gettimeofday(&end,NULL);
    time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
    
    printf("================ CPU COMPUTATION =================\n");
    printf("CPU Solve Time = %ld micro seconds\n", time);

    av_heats = (double*) malloc(sizeof(double)*n);
    
    gettimeofday(&start,NULL);
    find_av_heats(av_heats, &A1);
    gettimeofday(&end,NULL);

    time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
    printf("CPU Reduce Time = %ld micro seconds.\n",time);
    
  //  print_av_heats(av_heats,n);
  }
  
  initiate_matrix(&A2, n, m);
  double* gpu_av_heats = (double*) malloc(sizeof(double)*n);
  printf("================ GPU COMPUTATION =================\n");
  run_cuda_radiator(&A2,gpu_av_heats, p, t_flag);
//  print_av_heats(gpu_av_heats,n);
  if(show_av == 1)
  {
    int sig_errors = 0;
    for (i=0; i< A1.nRows; i++)
	for (j=0; j< A1.nColumns; j++)
	{
	  if (abs(A1.entry[i][j] - A2.entry[i][j])>0.00005)
	  {
	    sig_errors++;
	  }
	}
    printf("================ COMPARISON =================\n");
    printf("There were %d significant errors between CPU and GPU calculation.\n",sig_errors);
    free(av_heats);
    free_matrix(&A1);
  }

 
  free(gpu_av_heats);
  free_matrix(&A2);
  return 0;
}
