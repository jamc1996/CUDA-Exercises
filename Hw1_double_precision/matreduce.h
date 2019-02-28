#ifndef MATREDUCE_H
#define MATREDUCE_H

#ifdef __cplusplus
extern "C" 
{
	void run_time_tests(int n, int m, int seed, int block_size);
	void CPU_row_reduce(float* V, float** A, int n, int m);
	void CPU_col_reduce(float* V, float** A, int n, int m);
	void CPU_reduce_vec(float* tot, float* V, int n);
}
#endif

#endif
