#ifndef MATREDUCE_H
#define MATREDUCE_H

#ifdef __cplusplus
extern "C" 
{
	extern void write_times(int n, int m, int seed, int block_size);
	void run_basic_tests(int n, int m, int seed, int block_size);
	void run_time_tests(int n, int m, int seed, int block_size);
	void CPU_row_reduce(double* V, double** A, int n, int m);
	void CPU_col_reduce(double* V, double** A, int n, int m);
	void CPU_reduce_vec(double* tot, double* V, int n);
}
#endif

#endif
