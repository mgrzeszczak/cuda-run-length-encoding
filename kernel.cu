/******************************************
			INCLUDES
******************************************/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <cmath>

#include "utils/cuda_utils.h"
#include "utils/c_utils.h"
#include "utils/consts.h"

#include "tests/tests.h"

#include "kernels/scan.h"
#include "kernels/mask.h"
#include "kernels/compressed_mask.h"
#include "kernels/counts.h"
#include "kernels/symbols.h"

#include "rle/rle.h"
/******************************************
			MAIN FUNCTION
******************************************/
void run_tests() {
	test_scan(&cudaScan);
	test_mask(&cudaMask);
	test_compressed_mask(&cudaCompressedMask);
	test_counts(&cudaCounts);
	test_symbols(&cudaSymbols);
}

char* generate_data(int size, float compressability) {
	srand(time(0));
	int alphabet_length = 26;
	char *data = (char*)_malloc(sizeof(char)*size);
	data[0] = 'a' + rand() % alphabet_length;
	for (int i = 1; i < size; i++) {
		float r = randf();
		data[i] = (r <= compressability) ? (data[i - 1]) : ('a' + rand() % alphabet_length);
	}
	return data;
}

void measure_performance(void(*fun)(char*,int,char**,int**,int*),char* data, int size,char* label) {
	char *symbols;
	int length;
	int *runs;
	printf("\n%s\n", label);
	clock_t start = clock();
	fun(data, size, &symbols, &runs, &length);
	clock_t end = clock();
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("Time passed: %f s\n", seconds);
	printf("Output length: %d * 2\n", length);
	free(symbols);
	free(runs);
}

void run_comparison(int size, float compressability) {
	printf("Generating %d MB of data with %.2f %% compressability...\n", size/MB,compressability*100);
	char *data = generate_data(size,compressability);
	printf("Running performance tests...\n");
	measure_performance(&parallel_rle, data, size,"GPU VERSION");
	measure_performance(&cpu_rle, data, size,"CPU VERSION");
	printf("\nDone\n");
	free(data);
}

void read_arguments(int *sizeMb, float* compressability) {
	printf("Provide data size int megabytes (int):\n");
	scanf("%d", sizeMb);
	printf("Provide compressability (float): \n");
	scanf("%f", compressability);
}

int main()
{
	int sizeMb;
	float compressability;
	//run_tests();
	read_arguments(&sizeMb, &compressability);
	run_comparison(sizeMb*MB,compressability);
	return EXIT_SUCCESS;
}