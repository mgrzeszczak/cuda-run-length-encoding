#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>

#include "../utils/cuda_utils.h"
#include "../utils/c_utils.h"
#include "../utils/consts.h"

void cudaScan(int *arr, const int length, int** out);
__global__ void scanKernel(int *data, const int len, int *out);
__global__ void addPerBlockKernel(int *data, int *toAdd);

/* KERNEL FUNCTIONS */
__global__ void scanKernel(int *data, const int len, int *out)
{
	const int threadCount = 1024;
	const int log_1024 = 10;

	int blockCount = gridDim.x;
	int tbid = blockIdx.x*threadCount + threadIdx.x;
	int id = threadIdx.x;
	int offset = blockIdx.x * threadCount;
	// max id 65535 * 1024 length

	//int val = data[id];
	//__syncthreads();

	for (int d = 0; d < log_1024 - 1; d++) {
		if (id % (1 << (d + 1)) == 0) {
			data[offset + id + (1 << (d + 1)) - 1] = data[offset + id + (1 << d) - 1] + data[offset + id + (1 << (d + 1)) - 1];
		}
		__syncthreads();
	}

	if (id == 0)
		data[offset + 1024 - 1] = 0;
	__syncthreads();

	for (int d = log_1024 - 1; d >= 0; d--) {
		if (id % (1 << (d + 1)) == 0) {
			int tmp = data[offset + id + (1 << d) - 1];
			data[offset + id + (1 << d) - 1] = data[offset + id + (1 << (d + 1)) - 1];
			data[offset + id + (1 << (d + 1)) - 1] = tmp + data[offset + id + (1 << (d + 1)) - 1];
		}
		__syncthreads();
	}

	//data[id] += val;
}

__global__ void addPerBlockKernel(int *data, int *toAdd)
{
	const int threadCount = 1024;
	const int log_1024 = 10;

	int blockCount = gridDim.x;
	int tbid = blockIdx.x*threadCount + threadIdx.x;
	int id = threadIdx.x;
	int offset = blockIdx.x * threadCount;

	data[tbid] += toAdd[tbid / threadCount];
}

// CAN SERVE A MAX OF 65535 KB OF DATA
void cudaScan(int *data, const int length, int **out) {
	int *dev_data;

	*out = (int*)_malloc(sizeof(int)*length);
	int blockCount = ceil(((float)(length) / MAX_THREADS_PER_BLOCK));

	_cudaSetDevice(0);
	_cudaMalloc((void**)&dev_data, blockCount * MAX_THREADS_PER_BLOCK * sizeof(int));
	_cudaMemset((void*)dev_data, 0, blockCount * MAX_THREADS_PER_BLOCK *sizeof(int));

	_cudaMemcpy(dev_data, data, length * sizeof(int), cudaMemcpyHostToDevice);
	scanKernel << <blockCount, MAX_THREADS_PER_BLOCK >> >(dev_data, length, NULL);

	_cudaDeviceSynchronize("cudaScan");
	_cudaMemcpy(*out, dev_data, length * sizeof(int), cudaMemcpyDeviceToHost);

	if (blockCount > 1) {
		int *remaining = (int*)_malloc(sizeof(int)*blockCount);

		for (int i = 0; i < blockCount; i++) {
			if (i*MAX_THREADS_PER_BLOCK - 1 < length)
				remaining[i] = (*out)[(i + 1) * MAX_THREADS_PER_BLOCK - 1] + data[(i + 1) * MAX_THREADS_PER_BLOCK - 1];
			else remaining[i] = (*out)[(i + 1)*MAX_THREADS_PER_BLOCK - 1] + data[length - 1];
		}
		int *scanned;
		cudaScan(remaining, blockCount, &scanned);

		int *dev_scanned;
		_cudaMalloc((void**)&dev_scanned, blockCount * sizeof(int));
		_cudaMemset((void*)dev_data, 0, blockCount *sizeof(int));

		_cudaMemcpy(dev_scanned, scanned, blockCount * sizeof(int), cudaMemcpyHostToDevice);

		addPerBlockKernel << <blockCount, MAX_THREADS_PER_BLOCK >> >(dev_data, dev_scanned);

		_cudaMemcpy(*out, dev_data, length * sizeof(int), cudaMemcpyDeviceToHost);

		cudaFree(dev_scanned);

		free(scanned);
		free(remaining);
	}

	cudaFree(dev_data);
}