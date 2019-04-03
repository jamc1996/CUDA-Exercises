#ifndef PROPAGATE_H
#define PROPAGATE_H

#include "matrix.h"
#include "setmatrix.h"

/*	propagate.h -- header file for propagate.c
 *
 *	Author: John Cormican
 */


void iterative_solver(Matrix* A, int p);
void propagate_it(Matrix* A, Matrix* B);
void fill_B(Matrix* A, Matrix* B);
void find_av_heats(float* av_heats, Matrix* A);
void print_av_heats(float* av_heats, int n);


#endif
