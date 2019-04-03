#ifndef MATRIX_H
#define MATRIX_H

/*	matrix.h -- header file to store matrix struct for well organized data.
 *
 *   	Author: John Cormican
 *
 */


typedef struct
{
  float* data_;
  float** entry;
  int nRows;
  int nColumns;
}Matrix;

#endif
