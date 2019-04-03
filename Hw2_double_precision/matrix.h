#ifndef MATRIX_H
#define MATRIX_H

/*	matrix.h -- header file to store matrix struct for well organized data.
 *
 *   	Author: John Cormican
 *
 */


typedef struct
{
  double* data_;
  double** entry;
  int nRows;
  int nColumns;
}Matrix;

#endif
