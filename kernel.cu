/******************************************
			INCLUDES
******************************************/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
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

void rle(char *data, int length) {
	// for now data assumed < 64 MB
	int *mask;
	cudaMask(data, length, &mask);
	int *scannedMask;
	cudaScan(mask, length, &scannedMask, true);
	int *compressedMask;
	int compressedLength;
	cudaCompressedMask(scannedMask, length, &compressedMask, &compressedLength);

	int *counts;
	char* symbols;
	cudaCounts(compressedMask, compressedLength, &counts);
	cudaSymbols(compressedMask, compressedLength, data, length, &symbols);

	int outLength = compressedLength - 1;
	for (int i = 0; i < outLength; i++) {
		printf("%c x %d\n", symbols[i], counts[i]);
	}

	free(mask);
	free(scannedMask);
	free(compressedMask);
	free(counts);
	free(symbols);
}

void rle_large_data() {
	int size = 60 * MB;
	char *data = (char*)_malloc(sizeof(char) * size);
	for (int i = 0; i < size; i++) {
		data[i] = i > 30*MB? 'b' : 'a';
	}

	rle(data, size);
	free(data);
}

void rle_small_data() {
	char data[] = { 'a','a','a','a', 'b','b','b','b' };
	int length = 8;
	rle(data, length);
}

int main()
{
	//run_tests();
	//rle_small_data();
	rle_large_data();
    return 0;
}