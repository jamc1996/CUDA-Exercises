#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "setmatrix.h"
#include "propagate.h"

void iterative_solver(Matrix* A, int p)
{
  int i;
  Matrix B;
  allocate_empty(&B,A->nRows,A->nColumns);
  fill_B(&B,A); 
  for (i=0; i<p/2; i++)
  {
    propagate_it(&B,A);
    propagate_it(A,&B);
  }

  if(p%2 == 1)
  {
    propagate_it(&B,A);
  }
  free_matrix(&B);
}

void fill_B(Matrix* B, Matrix* A)
{
  int i,j;
  for (i=0; i<A->nRows; i++)
  {
    for (j=0; j<2; j++)
    {
      B->entry[i][j] = A->entry[i][j];	
    }
  }
}

void propagate_it(Matrix* A, Matrix* B)
{
  int i,j;
  for (i=0; i<A->nRows; i++)
  {
    for (j=2; j<A->nColumns; j++)
    {
	A->entry[i][j] = ((1.9*(B->entry[i][(j-2)%A->nColumns]))+(1.5*(B->entry[i][(j-1)%A->nColumns]))+ (0.5*(B->entry[i][(j+1)%A->nColumns])) + (0.1*(B->entry[i][(j+2)%A->nColumns])) + B->entry[i][j]);
	A->entry[i][j] /= (double)5.0;
    }
  }
}

void find_av_heats(double* av_heats, Matrix* A)
{
  int i,j;
  for (i=0;i<A->nRows;i++)
  {
    av_heats[i] = 0;
    for (j=0;j<A->nColumns;j++)
    {
      av_heats[i]+=A->entry[i][j];
    }
    av_heats[i] /= (double)A->nColumns;
  }
}

void print_av_heats(double* av_heats, int n)
{
  int i;
  printf("Average temperature of each row:\n");
  for (i=0;i<n;i++)
  {
    printf("Row[%d] -- %f\n",i,av_heats[i]);
  }
}















