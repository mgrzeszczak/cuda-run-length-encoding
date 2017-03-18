#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>

#include "../utils/cuda_utils.h"
#include "../utils/c_utils.h"
#include "../utils/consts.h"

__global__ void countsKernel(int *compressedMask, int *counts, int length);
void cudaCounts(int *compressedMask, int length, int **counts);

__global__ void countsKernel(int *compressedMask, int *counts, int length) {
	const int threadCount = 1024;
	const int log_1024 = 10;

	int blockCount = gridDim.x;
	int tbid = blockIdx.x*threadCount + threadIdx.x;
	int id = threadIdx.x;
	int offset = blockIdx.x * threadCount;

	if (tbid > length - 1) return;

	if (tbid > 0) {
		counts[tbid - 1] = compressedMask[tbid] - compressedMask[tbid - 1];
	}
}


void cudaCounts(int *compressedMask, int length, int **counts) {
	int *dev_data;
	int *dev_counts;

	int blockCount = ceil(((float)(length) / MAX_THREADS_PER_BLOCK));

	_cudaSetDevice(0);

	_cudaMalloc((void**)&dev_data, blockCount * MAX_THREADS_PER_BLOCK * sizeof(int));
	_cudaMemset((void*)dev_data, 0, blockCount * MAX_THREADS_PER_BLOCK *sizeof(int));
	_cudaMemcpy(dev_data, compressedMask, length * sizeof(int), cudaMemcpyHostToDevice);

	_cudaMalloc((void**)&dev_counts, blockCount * MAX_THREADS_PER_BLOCK * sizeof(int));

	countsKernel << <blockCount, MAX_THREADS_PER_BLOCK >> >(dev_data, dev_counts, length);

	_cudaDeviceSynchronize("cudaCounts");

	*counts = (int*)_malloc(sizeof(int)*(length - 1));

	_cudaMemcpy(*counts, dev_counts, (length-1) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_data);
	cudaFree(dev_counts);
}