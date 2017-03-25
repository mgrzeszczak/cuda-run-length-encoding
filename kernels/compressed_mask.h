#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>

#include "../utils/cuda_utils.h"
#include "../utils/c_utils.h"
#include "../utils/consts.h"

__global__ void compressedMaskKernel(int *mask, int *compressedMask, int *compressedLength, int length);
void cudaCompressedMask(int *mask, const int length, int **compressedMask, int *compressedLength);

__global__ void compressedMaskKernel(int *mask, int *compressedMask, int *compressedLength, int length) {
	const int threadCount = 1024;
	const int log_1024 = 10;

	int blockCount = gridDim.x;
	int tbid = blockIdx.x*threadCount + threadIdx.x;
	int id = threadIdx.x;
	int offset = blockIdx.x * threadCount;

	if (tbid >= length) return;

	while (tbid < length) {
		if (tbid == 0) {
			compressedMask[mask[tbid] - 1] = 0;
		}
		else if (mask[tbid - 1] != mask[tbid]) {
			compressedMask[mask[tbid] - 1] = tbid;
		}

		if (tbid == length - 1) {
			compressedMask[mask[tbid]] = length;
			*compressedLength = mask[tbid] + 1;
		}
		tbid += blockCount*threadCount;
	}
}

// CAN SERVE A MAX OF 65535 KB OF DATA
void cudaCompressedMask(int *mask, const int length, int **compressedMask, int *compressedLength) {
	int *dev_mask;
	int *dev_compressedMask;
	int *dev_compressedLength;

	int blockCount = ceil(((float)(length) / MAX_THREADS_PER_BLOCK));

	_cudaSetDevice(0);

	_cudaMalloc((void**)&dev_mask, blockCount * MAX_THREADS_PER_BLOCK * sizeof(int));
	_cudaMemset((void*)dev_mask, 0, blockCount * MAX_THREADS_PER_BLOCK *sizeof(int));
	_cudaMemcpy(dev_mask, mask, length * sizeof(int), cudaMemcpyHostToDevice);

	_cudaMalloc((void**)&dev_compressedMask, blockCount * MAX_THREADS_PER_BLOCK * sizeof(int));
	_cudaMalloc((void**)&dev_compressedLength, sizeof(int));

	compressedMaskKernel << <blockCount, MAX_THREADS_PER_BLOCK >> >(dev_mask,dev_compressedMask,dev_compressedLength,length);

	_cudaDeviceSynchronize("cudaCompressedMask");

	_cudaMemcpy(compressedLength, dev_compressedLength, sizeof(int), cudaMemcpyDeviceToHost);
	*compressedMask = (int*)_malloc(sizeof(int)*(*compressedLength));
	_cudaMemcpy(*compressedMask, dev_compressedMask, (*compressedLength) * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_compressedMask);
	cudaFree(dev_mask);
	cudaFree(dev_compressedLength);
}