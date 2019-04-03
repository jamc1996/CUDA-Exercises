#ifndef SETMATRIX_H
#define SETMATRIX_H

/*	setmatrix.h -- header file for setmatrix.c
 *
 *	Author: John Cormican
 */

#include "matrix.h"

void initiate_matrix(Matrix* A,int n, int m);
void allocate_empty(Matrix* A, int n, int m);
void set_boundary(Matrix* A);
void print_matrix(Matrix* A);
void free_matrix(Matrix* A);


#endif
