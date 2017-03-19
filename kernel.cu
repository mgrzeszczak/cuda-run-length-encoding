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

void rle(char *data, int length, bool debug) {
	int inputLength = length;

	char *outSymbols = NULL;
	int *outCounts = NULL;

	int outLength =0;

	int part = 1;

	while (length > 60 * MB) {
		int *mask;
		cudaMask(data, 60*MB, &mask);
		int *scannedMask;
		cudaScan(mask, 60 * MB, &scannedMask, true);
		int *compressedMask;
		int compressedLength;
		cudaCompressedMask(scannedMask, 60 * MB, &compressedMask, &compressedLength);

		free(mask);
		free(scannedMask);

		int *counts;
		char* symbols;
		cudaCounts(compressedMask, compressedLength, &counts);
		cudaSymbols(compressedMask, compressedLength, data, 60 * MB, &symbols);
		int partLength = compressedLength - 1;
		free(compressedMask);

		if (debug) {
			printf("Part results \n");
			for (int i = 0; i < partLength; i++) {
				printf("%c x %d\n", symbols[i], counts[i]);
			}
			printf("\n");
		}
		

		int prevLength = outLength;
		outLength += partLength;
		if (outSymbols == NULL) {
			outSymbols = (char*)_malloc(sizeof(char) * partLength);
			outCounts = (int*)_malloc(sizeof(int) * partLength);
		}
		else {
			if (outSymbols[prevLength - 1] == symbols[0]) {
				outLength--;
			}
			outSymbols = (char*)_realloc(outSymbols, sizeof(char) * outLength);
			outCounts = (int*)_realloc(outCounts, sizeof(int) * outLength);
		}

		int offset = 0;
		if (outSymbols[prevLength - 1] == symbols[0]) {
			outCounts[prevLength - 1] += counts[0];
			offset = 1;
		}
		memcpy(outSymbols + prevLength, symbols + offset, partLength);
		memcpy(outCounts + prevLength, counts + offset, sizeof(int)*partLength);

		length -= 60 * MB;
		data += 60 * MB;
		
		
		free(counts);
		free(symbols);

		printf("%d MB processed...\n",60*part);
		part++;
	}

	int *mask;
	cudaMask(data, length, &mask);
	int *scannedMask;
	cudaScan(mask, length, &scannedMask, true);
	int *compressedMask;
	int compressedLength;
	cudaCompressedMask(scannedMask, length, &compressedMask, &compressedLength);

	free(mask);
	free(scannedMask);

	int *counts;
	char* symbols;
	cudaCounts(compressedMask, compressedLength, &counts);
	cudaSymbols(compressedMask, compressedLength, data, length, &symbols);

	free(compressedMask);

	int partLength = compressedLength - 1;
	int prevLength = outLength;

	if (debug) {
		printf("Part results \n");
		for (int i = 0; i < partLength; i++) {
			printf("%c x %d\n", symbols[i], counts[i]);
		}
		printf("\n");
	}
	

	outLength += partLength;
	if (outSymbols == NULL) {
		outSymbols = (char*)_malloc(sizeof(char) * partLength);
		outCounts = (int*)_malloc(sizeof(int) * partLength);
	}
	else {
		if (outSymbols[prevLength - 1] == symbols[0]) {
			outLength--;
		}
		outSymbols = (char*)_realloc(outSymbols, sizeof(char) * outLength);
		outCounts = (int*)_realloc(outCounts, sizeof(int) * outLength);
	}

	int offset = 0;
	if (outSymbols[prevLength - 1] == symbols[0]) {
		outCounts[prevLength - 1] += counts[0];
		offset = 1;
	}
	memcpy(outSymbols + prevLength, symbols+offset, partLength);
	memcpy(outCounts + prevLength, counts+offset, sizeof(int)*partLength);

	if (debug) {
		for (int i = 0; i < outLength; i++) {
			printf("%c x %d\n", outSymbols[i], outCounts[i]);
		}
	}

	printf("Input length: %d\n", inputLength);
	printf("Compressed length: %d\n", outLength);

	
	
	free(counts);
	free(symbols);
	free(outCounts);
	free(outSymbols);
}

void cpu_rle(char *data, int length) {
	char *compressed = (char*)_malloc(sizeof(char)*length);
	int *lengths = (int*)_malloc(sizeof(int)*length);

	char c = data[0];
	int count = 1;
	int position = 0;
	for (int i = 1; i < length; i++) {

		if (data[i] == c) {
			count++;
			continue;
		}

		compressed[position] = c;
		lengths[position] = count;

		count = 1;
		c = data[i];
		position++;
	}

	printf("CPU compressed length: %d\n", position + 1);

	free(compressed);
	free(lengths);
}

void rle_large_data(int megabytes) {
	srand(123);
	int size = megabytes * MB;
	char *data = (char*)_malloc(sizeof(char) * size);

	printf("Generating %d MB of data...\n", megabytes);
	for (int i = 0; i < size; i++) {
		//data[i] = i >= size/2 ? 'b' : 'a';
		data[i] = rand() % 4 + 'a';
	}

	/*
	int count = 1;
	char c = 'a';
	for (int i = 1; i < size; i++) {
		if (data[i] != c) {
			printf("%c --> %d\n", c, count);
			c = data[i];
			count = 1;
		}
		else {
			count++;
		}
	}
	printf("%c --> %d\n", c, count);*/

	printf("Running parallel RLE...\n");
	rle(data, size, false);


	cpu_rle(data, size);

	free(data);
}

void rle_small_data() {

	int length = 1040;
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
}

void rle_gpu_large(int megabytes) {
	srand(123);
	int size = megabytes * MB;
	char *data = (char*)_malloc(sizeof(char) * size);

	printf("Generating %d MB of data...\n", megabytes);
	for (int i = 0; i < size; i++) {
		data[i] = rand() % 4 + 'a';
	}

	printf("Running parallel RLE...\n");


	char *symbols;
	int out_length;
	int *runs;
	parallel_rle(data, size, &symbols, &runs, &out_length);

	printf("Compressed size: %d\n", out_length);

	cpu_rle(data, size);
	free(data);
}

int main()
{
	//run_tests();
	//rle_small_data();
	//rle_large_data(100);
	rle_gpu_large(50);
    return 0;
}