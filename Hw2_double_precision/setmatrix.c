#include <stdio.h>
#include <stdlib.h>

#include "setmatrix.h"

/* 	setmatrix.c -- program to 

*/

void initiate_matrix(Matrix *A, int n, int m)
{
  int i;
  allocate_empty(A,n,m);

  for (i=0;i<A->nRows*A->nColumns;i++)
  {
    A->data_[i] = 0.0;
  }
  set_boundary(A);
//  print_matrix(A);
}

void set_boundary(Matrix* A)
{
  int i;
  for (i=0;i<(A->nRows);i++)
  {
    A->entry[i][0] = 1.00*(double)(i+1)/(double)(A->nRows);
    A->entry[i][1] = 0.8*(double)(i+1)/(double)(A->nRows);
  }
}

void print_matrix(Matrix* A)
{
  int i,j;
  for (i=0;i<A->nRows;i++)
  {
    for (j=0;j<A->nColumns;j++)
    {
      printf(" %.3f ",A->entry[i][j]);
    }
    printf("\n");
  }
}

void allocate_empty(Matrix* A, int n, int m)
{
  int i;
  A->nRows = n;
  A->nColumns = m;
  A->data_ = (double*) malloc(A->nRows*A->nColumns*sizeof(double));
  A->entry = (double**) malloc(A->nRows*sizeof(double*));
  for (i=0;i<A->nRows;i++)
  {
    A->entry[i] = &(A->data_[i*A->nColumns]);
  }  
}

void free_matrix(Matrix* A)
{
  free(A->data_);
  free(A->entry);
}





