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

void cpu_rle(char *data, int length, char** compressed, int** lengths, int *out_length) {
	*compressed = (char*)_malloc(sizeof(char)*length);
	*lengths = (int*)_malloc(sizeof(int)*length);

	char c = data[0];
	int count = 1;
	int position = 0;
	for (int i = 1; i < length; i++) {

		if (data[i] == c) {
			count++;
			continue;
		}

		(*compressed)[position] = c;
		(*lengths)[position] = count;

		count = 1;
		c = data[i];
		position++;
	}

	*out_length = position + 1;
}

void rle_small_data() {
	int length = 10*MB;
	char *data = (char*)_malloc(sizeof(char)*length);
	for (int i = 0; i < length; i++)  data[i] = 'a';

	//char data[] = { 'a','a','a','a', 'b','b','b','b' };
	//int length = 8;
	//rle(data, length, true);

	int out_length;
	char *symbols;
	int *runs;

	parallel_rle(data, length, &symbols, &runs, &out_length);

	for (int i = 0; i < out_length; i++) {
		printf("%c x %d\n", symbols[i], runs[i]);
	}
	free(data);
}

void rle_large_data(int megabytes) {
	srand(123);
	int size = megabytes * MB;
	char *data = (char*)_malloc(sizeof(char) * size);

	printf("Generating %d MB of data...\n", megabytes);
	/*for (int i = 0; i < size; i++) {
		data[i] = rand() % 4 + 'a';
	}*/
	memset(data, 'a', size);
	
	char *g_symbols;
	int g_length;
	int *g_runs;

	printf("Running GPU RLE...\n");
	parallel_rle(data, size, &g_symbols, &g_runs, &g_length);
	printf("GPU Compressed size: %d\n", g_length);
	
	char *c_symbols;
	int *c_runs;
	int c_length;

	printf("Running CPU RLE...\n");
	cpu_rle(data, size,&c_symbols,&c_runs,&c_length);
	printf("CPU Compressed size: %d\n", c_length);


	free(data);
	free(g_symbols);
	free(g_runs);
	free(c_symbols);
	free(c_runs);
}

int main()
{
	//run_tests();
	//rle_small_data();
	rle_large_data(500);
	return EXIT_SUCCESS;
}