#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>

#include "../utils/cuda_utils.h"
#include "../utils/c_utils.h"
#include "../utils/consts.h"

__global__ void symbolsKernel(int *compressedMask, char *data, int maskLength, char *symbols);
void cudaSymbols(int *compressedMask, int maskLength, char *data, int dataLength, char **symbols);

__global__ void symbolsKernel(int *compressedMask, char *data, int maskLength, char *symbols) {
	const int threadCount = 1024;
	const int log_1024 = 10;

	int blockCount = gridDim.x;
	int tbid = blockIdx.x*threadCount + threadIdx.x;
	int id = threadIdx.x;
	int offset = blockIdx.x * threadCount;

	if (tbid > maskLength - 1) return;
	
	while (tbid < maskLength) {
		if (tbid < maskLength - 1) {
			symbols[tbid] = data[compressedMask[tbid]];
		}
		tbid += blockCount*threadCount;
	}
	
}


void cudaSymbols(int *compressedMask, int maskLength, char *data, int dataLength, char **symbols) {
	int *dev_mask;
	char *dev_data;
	char *dev_symbols;

	int blockCount = ceil(((float)(maskLength) / MAX_THREADS_PER_BLOCK));

	_cudaSetDevice(0);

	_cudaMalloc((void**)&dev_mask, blockCount * MAX_THREADS_PER_BLOCK * sizeof(int));
	_cudaMemcpy(dev_mask, compressedMask, maskLength * sizeof(int), cudaMemcpyHostToDevice);

	_cudaMalloc((void**)&dev_data, dataLength * sizeof(char));
	_cudaMemcpy(dev_data, data, dataLength * sizeof(char), cudaMemcpyHostToDevice);

	_cudaMalloc((void**)&dev_symbols, blockCount * MAX_THREADS_PER_BLOCK * sizeof(char));

	symbolsKernel << <blockCount, MAX_THREADS_PER_BLOCK >> >(dev_mask, dev_data, maskLength, dev_symbols);

	_cudaDeviceSynchronize("cudaSymbols");

	*symbols = (char*)_malloc(sizeof(char)*(maskLength - 1));

	_cudaMemcpy(*symbols, dev_symbols, (maskLength-1) * sizeof(char), cudaMemcpyDeviceToHost);

	cudaFree(dev_data);
	cudaFree(dev_mask);
}