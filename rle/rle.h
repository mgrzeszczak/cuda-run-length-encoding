#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>

#include "../utils/cuda_utils.h"
#include "../utils/c_utils.h"
#include "../utils/consts.h"

#include "../kernels/compressed_mask.h"
#include "../kernels/mask.h"
#include "../kernels/scan.h"
#include "../kernels/symbols.h"
#include "../kernels/counts.h"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

//using namespace thrust;

void parallel_rle(char *data, int length, char **symbols, int** runs, int *out_length);
void rle_batch(char *data, int length, char *symbols, int* runs, int *out_length);
void cpu_rle(char *data, int length, char** compressed, int** lengths, int *out_length);

void rle_batch(char *data, int length, char *symbols, int* runs, int *out_length) {
	// GPU MEMORY
	char* d_data;
	int *d_mask;
	int *d_compressed_mask;
	int *d_compressed_length;
	char *d_symbols;
	int *d_runs;

	//_cudaSetDevice(0);

	int blockCount = ceil(((float)(length) / MAX_THREADS_PER_BLOCK));
	if (blockCount > MAX_BLOCKS_PER_GRID) blockCount = MAX_BLOCKS_PER_GRID;
	

	int arrLen = blockCount*MAX_THREADS_PER_BLOCK;
	if (arrLen < length) arrLen = length;

	//_cudaPrintMemory();
	// allocate gpu memory
	
	_cudaMalloc((void**)&d_data, arrLen*sizeof(char)); // 150MB
	_cudaMalloc((void**)&d_mask, arrLen*sizeof(int)); // 600MB

	// total 750 MB
	//_cudaPrintMemory();

	_cudaMemcpy(d_data, data, length * sizeof(char), cudaMemcpyHostToDevice);

	// CALCULATE MASK
	maskKernel << <blockCount, MAX_THREADS_PER_BLOCK >> >(d_data, d_mask,length);
	_cudaDeviceSynchronize("cudaMask");

	// SCAN MASK
	//cudaScanGpuMem(d_mask, length, true);
	thrust::inclusive_scan(thrust::device, d_mask, d_mask + length, d_mask);
	

	//thrust::inclusive_scan(d_mask, d_mask+length, d_mask); // in-place scan

	
	/*
	int *c_data = (int*)_malloc(sizeof(int)*length);
	_cudaMemcpy(c_data, d_mask, sizeof(int)*length, cudaMemcpyDeviceToHost);

	printf("Scanned\n");
	for (int i = 0; i < length; i++) {
	printf("%d ", c_data[i]);
	}
	printf("\n");
	*/


	/* MEMORY HEAVY SCAN */
	/*
	int *c_data = (int*)_malloc(sizeof(int)*length);
	_cudaMemcpy(c_data, d_mask, sizeof(int)*length, cudaMemcpyDeviceToHost);

	printf("Mask\n");
	for (int i = 0; i < length; i++) {
	printf("%d ", c_data[i]);
	}
	printf("\n");

	int *c_scanned;
	cudaScan(c_data, length, &c_scanned, true);

	printf("Scanned\n");
	for (int i = 0; i < length; i++) {
	printf("%d ", c_scanned[i]);
	}
	printf("\n");

	_cudaMemcpy(d_mask, c_scanned, sizeof(int)*length, cudaMemcpyHostToDevice);
	*/
	//_cudaPrintMemory();
	// COMPRESS MASK
	_cudaMalloc((void**)&d_compressed_mask, arrLen*sizeof(int)); // 600MB
	_cudaMalloc((void**)&d_compressed_length, sizeof(int));

	// total 1350MB - max
	//_cudaPrintMemory();
	compressedMaskKernel << <blockCount, MAX_THREADS_PER_BLOCK >> >(d_mask, d_compressed_mask, d_compressed_length, length);
	_cudaDeviceSynchronize("cudaCompressedMask");

	cudaFree(d_mask);
	// total 750MB
	//_cudaPrintMemory();

	int compressedLength;
	_cudaMemcpy(&compressedLength, d_compressed_length, sizeof(int), cudaMemcpyDeviceToHost);

	_cudaMalloc((void**)&d_symbols, (compressedLength - 1)*sizeof(char));
	_cudaMalloc((void**)&d_runs, (compressedLength - 1)*sizeof(int));

	// SYMBOLS OUT
	symbolsKernel << <blockCount, MAX_THREADS_PER_BLOCK >> >(d_compressed_mask, d_data, compressedLength, d_symbols);
	_cudaDeviceSynchronize("cudaSymbols");

	// RUNS OUT
	countsKernel << <blockCount, MAX_THREADS_PER_BLOCK >> >(d_compressed_mask, d_runs, compressedLength);
	_cudaDeviceSynchronize("cudaCounts");

	// FREE MEMORY
	cudaFree(d_data);
	cudaFree(d_compressed_length);
	cudaFree(d_compressed_mask);

	*out_length = compressedLength - 1;
	//*symbols = (char*)_malloc(sizeof(char)*(*out_length));
	//*runs = (int*)_malloc(sizeof(int)*(*out_length));

	_cudaMemcpy(symbols, d_symbols, sizeof(char)*(*out_length), cudaMemcpyDeviceToHost);
	_cudaMemcpy(runs, d_runs, sizeof(int)*(*out_length), cudaMemcpyDeviceToHost);

	cudaFree(d_runs);
	cudaFree(d_symbols);
	//_cudaPrintMemory();
}

void parallel_rle(char *data, int length, char **symbols, int** runs, int *out_length) {
	//size_t mem_free;
	//size_t mem_total;
	//cudaMemGetInfo(&mem_free, &mem_total);
	//mem_free /= MB;
	//mem_total /= MB;
	long step = 500*MB;
	//long step = ((long)mem_free) / 10 * MB;
	if (step > length) step = length;
	long stepMB = step / MB;
	//printf("Chunk size: %d MB\n",stepMB);

	int inputLength = length;

	char *out_symbols;
	int *out_runs;
	int processed_length = 0;

	out_symbols = (char*)_malloc(sizeof(char)*length);
	out_runs = (int*)_malloc(sizeof(int)*length);
	
	char *part_symbols = (char*)_malloc(sizeof(char)*step);
	int *part_runs = (int*)_malloc(sizeof(int)*step);
	int part_length;
	while (length > step) {

		rle_batch(data, step, part_symbols, part_runs, &part_length);

		int position = processed_length;
		processed_length += part_length;
		int offset = 0;
		if (position > 0 && out_symbols[position - 1] == part_symbols[0]) {
			out_runs[position - 1] += part_runs[0];
			offset = 1;
			processed_length -= 1;
			part_length -= 1;
		}

		memcpy(out_symbols + position, part_symbols + offset, sizeof(char)*part_length);
		memcpy(out_runs + position, part_runs + offset, sizeof(int)*part_length);

		length -= step;
		data += step;

	}

	rle_batch(data, length, part_symbols, part_runs, &part_length);

	int position = processed_length;
	processed_length += part_length;
	int offset = 0;

	if (position > 0 && out_symbols[position - 1] == part_symbols[0]) {
		out_runs[position - 1] += part_runs[0];
		offset = 1;
		processed_length -= 1;
		part_length -= 1;
	}

	memcpy(out_symbols + position, part_symbols + offset, sizeof(char)*part_length);
	memcpy(out_runs + position, part_runs + offset, sizeof(int)*part_length);

	free(part_symbols);
	free(part_runs);

	*out_length = processed_length;
	*symbols = (char*)_realloc(out_symbols, sizeof(char) * processed_length);
	*runs = (int*)_realloc(out_runs, sizeof(int) * processed_length);
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
