#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>

#include "../utils/cuda_utils.h"
#include "../utils/c_utils.h"
#include "../utils/consts.h"

void cudaScan(int *arr, const int length, int** out, bool inclusive);
__global__ void scanKernel(int *data, const int len, int *out);
__global__ void addPerBlockKernel(int *data, int *toAdd);
__global__ void addVectorsKernel(int *data, int *toAdd, int length);

__global__ void remaindersKernel(int *data, int *remainders, int *copy, int length);
void test_print(int *d_data, int length, char* s);

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

__global__ void addVectorsKernel(int *data, int *toAdd, int length) {
	const int threadCount = 1024;
	const int log_1024 = 10;

	int blockCount = gridDim.x;
	int tbid = blockIdx.x*threadCount + threadIdx.x;
	int id = threadIdx.x;
	int offset = blockIdx.x * threadCount;

	if (tbid >= length) return;
	data[tbid] += toAdd[tbid];
}

__global__ void remaindersKernel(int *data, int *remainders, int *copy, int length) {
	const int threadCount = 1024;
	const int log_1024 = 10;

	int blockCount = gridDim.x;
	int tbid = blockIdx.x*threadCount + threadIdx.x;
	int id = threadIdx.x;
	int offset = blockIdx.x * threadCount;

	if (tbid >= length) return;

	if ((tbid+1)%threadCount == 0) {
		remainders[(tbid+1) / (threadCount) -1] = data[tbid] + copy[tbid];
	}

	if (tbid == length - 1) {
		remainders[tbid / threadCount] = copy[tbid] + data[tbid];
	}
}

// CAN SERVE A MAX OF 65535 KB OF DATA
void cudaScan(int *data, const int length, int **out, bool inclusive) {
	int *dev_data;

	*out = (int*)_malloc(sizeof(int)*length);
	int blockCount = ceil(((float)(length) / MAX_THREADS_PER_BLOCK));

	_cudaSetDevice(0);
	_cudaMalloc((void**)&dev_data, blockCount * MAX_THREADS_PER_BLOCK * sizeof(int));
	_cudaMemset((void*)dev_data, 0, blockCount * MAX_THREADS_PER_BLOCK *sizeof(int));

	_cudaMemcpy(dev_data, data, length * sizeof(int), cudaMemcpyHostToDevice);
	scanKernel << <blockCount, MAX_THREADS_PER_BLOCK >> >(dev_data, length, NULL);

	test_print(dev_data, length, "Scanned data");

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
		cudaScan(remaining, blockCount, &scanned,false);

		int *dev_scanned;
		_cudaMalloc((void**)&dev_scanned, blockCount * sizeof(int));
		_cudaMemset((void*)dev_scanned, 0, blockCount *sizeof(int));
		_cudaMemcpy(dev_scanned, scanned, blockCount * sizeof(int), cudaMemcpyHostToDevice);

		addPerBlockKernel << <blockCount, MAX_THREADS_PER_BLOCK >> >(dev_data, dev_scanned);

		

		cudaFree(dev_scanned);

		free(scanned);
		free(remaining);
	}

	if (inclusive) {
		int *values;
		_cudaMalloc((void**)&values, blockCount * MAX_THREADS_PER_BLOCK *sizeof(int));
		_cudaMemset((void*)values, 0, blockCount * MAX_THREADS_PER_BLOCK *sizeof(int));
		_cudaMemcpy(values, data, length * sizeof(int), cudaMemcpyHostToDevice);

		addVectorsKernel << <blockCount,MAX_THREADS_PER_BLOCK >> >(dev_data,values,length);

		cudaFree(values);
	}

	_cudaMemcpy(*out, dev_data, length * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_data);
}

void cudaScanGpuMem(int *d_data, const int length, bool inclusive) {
	int *d_copy;
	int blockCount = ceil(((float)(length) / MAX_THREADS_PER_BLOCK));

	_cudaSetDevice(0);
	_cudaMalloc((void**)&d_copy, MAX_THREADS_PER_BLOCK*blockCount*sizeof(int));
	_cudaMemcpy(d_copy, d_data, sizeof(int)*length, cudaMemcpyDeviceToDevice);

	test_print(d_data, length, "Initial data:");
	
	scanKernel << <blockCount, MAX_THREADS_PER_BLOCK >> >(d_data, length, NULL);
	_cudaDeviceSynchronize("cudaScan");

	test_print(d_data, length, "Scanned data");

	if (blockCount > 1) {
		int *d_remaining;
		_cudaMalloc((void**)&d_remaining, blockCount*sizeof(int));

		remaindersKernel << <blockCount, MAX_THREADS_PER_BLOCK >> >(d_data, d_remaining, d_copy, length);
		_cudaDeviceSynchronize("remaindersKernel");

		test_print(d_remaining, blockCount, "Remaining values");

		cudaScanGpuMem(d_remaining, blockCount, false);

		test_print(d_remaining, blockCount, "Scanned remaining values");

		addPerBlockKernel << <blockCount, MAX_THREADS_PER_BLOCK >> >(d_data, d_remaining);
		_cudaDeviceSynchronize("addPerBlockKernel");

		cudaFree(d_remaining);
	}

	if (inclusive) {
		test_print(d_copy, length, "Copy:");
		addVectorsKernel << <blockCount, MAX_THREADS_PER_BLOCK >> >(d_data, d_copy, length);
		_cudaDeviceSynchronize("addVectorsKernel");

		test_print(d_data, length, "Added vectors:");
	}
	cudaFree(d_copy);
}

void test_print(int *d_data, int length, char* s) {
	if (!DEBUG) return;
	int *c_data = (int*)_malloc(sizeof(int)*length);
	_cudaMemcpy(c_data, d_data, sizeof(int)*length, cudaMemcpyDeviceToHost);
	printf("%s\n", s);
	for (int i = 0; i < length; i++) {
		printf("%d ", c_data[i]);
	}
	printf("\n");
	free(c_data);
}