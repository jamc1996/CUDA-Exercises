#include <stdio.h>
#include <stdlib.h>

#include "setmatrix.h"

/* 	setmatrix.c -- program to perform useful operation in matrix set up.
 *
 *	Author: John Cormican 
 *
 *	Purpouse: To allow initialisation, freeing, printing, etc. of matrix struct
 *
 *	Usage: Various function called from main.c
 */

void initiate_matrix(Matrix *A, int n, int m)
/* Functioin to allocate and fill matrix appropriately. */
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
/* Function to set the boundary conditions. */
{
  int i;
  for (i=0;i<(A->nRows);i++)
  {
    A->entry[i][0] = 1.00*(float)(i+1)/(float)(A->nRows);
    A->entry[i][1] = 0.8*(float)(i+1)/(float)(A->nRows);
  }
}

void print_matrix(Matrix* A)
/* Function to print Matrix A*/
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
/* Function to allocate an emptry nxm Matrix. */
{
  int i;
  A->nRows = n;
  A->nColumns = m;
  A->data_ = (float*) malloc(A->nRows*A->nColumns*sizeof(float));
  A->entry = (float**) malloc(A->nRows*sizeof(float*));
  for (i=0;i<A->nRows;i++)
  {
    A->entry[i] = &(A->data_[i*A->nColumns]);
  }  
}

void free_matrix(Matrix* A)
/* Function to free dynamically allocated memory of A. */
{
  free(A->data_);
  free(A->entry);
}





