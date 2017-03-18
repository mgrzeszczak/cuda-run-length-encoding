#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include "consts.h"

void _cudaSetDevice(int device) {
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		ERR("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
}

void _cudaMalloc(void **dest, size_t size) {
	cudaError_t cudaStatus = cudaMalloc(dest, size);
	if (cudaStatus != cudaSuccess) {
		ERR("cudaMalloc failed!");
	}
}

void _cudaMemset(void *dest, int val, size_t size) {
	cudaError_t cudaStatus = cudaMemset(dest, 0, size);
	if (cudaStatus != cudaSuccess) {
		ERR("cudaMalloc failed!");
	}
}

void _cudaMemcpy(void *dest, const void *src, size_t size, cudaMemcpyKind kind) {
	cudaError_t cudaStatus = cudaMemcpy(dest, src, size, kind);
	if (cudaStatus != cudaSuccess) {
		ERR("cudaMemcpy failed!");
	}
}

void _cudaDeviceSynchronize() {
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		ERR("scan launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		ERR("cudaDeviceSynchronize returned error code %d after launching scan!\n", cudaStatus);
	}
}
