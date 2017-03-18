#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>

#include "../utils/cuda_utils.h"
#include "../utils/c_utils.h"
#include "../utils/consts.h"

__global__ void maskKernel(int *data, int *mask);
void cudaMask(int *data, const int length, int **mask);

__global__ void maskKernel(int *data, int *mask) {
	const int threadCount = 1024;
	const int log_1024 = 10;

	int blockCount = gridDim.x;
	int tbid = blockIdx.x*threadCount + threadIdx.x;
	int id = threadIdx.x;
	int offset = blockIdx.x * threadCount;

	if (tbid == 0) mask[tbid] = 1;
	else {
		mask[tbid] = !(data[tbid] == data[tbid - 1]);
	}
}

// CAN SERVE A MAX OF 65535 KB OF DATA
void cudaMask(int *data, const int length, int **mask) {
	int *dev_data;

	*mask = (int*)_malloc(sizeof(int)*length);
	int blockCount = ceil(((float)(length) / MAX_THREADS_PER_BLOCK));

	_cudaSetDevice(0);
	_cudaMalloc((void**)&dev_data, blockCount * MAX_THREADS_PER_BLOCK * sizeof(int));
	_cudaMemset((void*)dev_data, 0, blockCount * MAX_THREADS_PER_BLOCK *sizeof(int));
	_cudaMemcpy(dev_data, data, length * sizeof(int), cudaMemcpyHostToDevice);

	maskKernel << <blockCount, MAX_THREADS_PER_BLOCK >> >(dev_data, NULL);

	_cudaDeviceSynchronize();
	_cudaMemcpy(*mask, dev_data, length * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_data);
}