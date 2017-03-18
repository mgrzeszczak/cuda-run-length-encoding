/******************************************
			INCLUDES
******************************************/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>

#include "cuda_utils.h"
#include "c_utils.h"
#include "consts.h"
#include "tests.h"
/******************************************
			FUNCTION DECLARATIONS
******************************************/
__global__ void scanKernel(int *data, const int len, int *out);
__global__ void addPerBlockKernel(int *data, int *toAdd);

__global__ void maskKernel(int *data, int *mask);
__global__ void compressedMaskKernel(int *mask, int *compressedMask);
__global__ void compressedMaskKernel(int *data, int *compressedMask, int* runLengths, int* runSymbols);

cudaError_t cudaScan(int *arr, const int length, int** out);

/******************************************
			FUNCTION CODE
******************************************/

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

	for (int d = 0; d < log_1024-1; d++) {
		if (id % (1 << (d + 1)) == 0) {
			data[offset+id + (1<<(d+1)) - 1] = data[offset+id + (1<<d) - 1] + data[offset+id + (1<<(d+1)) - 1];
		}
		__syncthreads();
	}

	if (id == 0)
		data[offset+1024 - 1] = 0;
	__syncthreads();

	for (int d = log_1024 - 1; d >= 0; d--) {
		if (id % (1<<(d+1)) == 0) {
			int tmp = data[offset+id + (1<<d) - 1];
			data[offset+id + (1<<d) - 1] = data[offset+id + (1<<(d+1)) - 1];
			data[offset+id + (1<<(d+1)) - 1] = tmp + data[offset+id + (1<<(d+1)) - 1];
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
__global__ void compressedMaskKernel(int *mask, int *compressedMask) {

}
__global__ void compressedMaskKernel(int *data, int *compressedMask, int* runLengths, int* runSymbols) {

}

/* HELPER CUDA FUNCTIONS */



/* KERNEL WRAPPING FUNCTIONS */

// CAN SERVE A MAX OF 65535 KB OF DATA
void cudaMask(int *data, const int length, int **mask) {
	int *dev_data;

	*mask = (int*)_malloc(sizeof(int)*length);
	int blockCount = ceil(((float)(length) / MAX_THREADS_PER_BLOCK));

	_cudaSetDevice(0);
	_cudaMalloc((void**)&dev_data, blockCount * MAX_THREADS_PER_BLOCK * sizeof(int));
	_cudaMemset((void*)dev_data, 0, blockCount * MAX_THREADS_PER_BLOCK *sizeof(int));
	_cudaMemcpy(dev_data, data, length * sizeof(int), cudaMemcpyHostToDevice);

	maskKernel << <blockCount, MAX_THREADS_PER_BLOCK >> >(dev_data,NULL);

	_cudaDeviceSynchronize();
	_cudaMemcpy(*mask, dev_data, length * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_data);
}


// CAN SERVE A MAX OF 65535 KB OF DATA
cudaError_t cudaScan(int *data, const int length, int **out) {
	int *dev_data;
	cudaError_t cudaStatus;

	//int count = 0;
	//for (int i = 0; i < length; i++) count++;
	//	printf("counted %d\n\n", count);

	*out = (int*)_malloc(sizeof(int)*length);
	if (*out == NULL) ERR("Failed to malloc out cudaScan");

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	int blockCount = ceil(((float)(length) / MAX_THREADS_PER_BLOCK));

	cudaStatus = cudaMalloc((void**)&dev_data, blockCount * MAX_THREADS_PER_BLOCK * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaMemset((void*)dev_data, 0, blockCount * MAX_THREADS_PER_BLOCK *sizeof(int));

	cudaStatus = cudaMemcpy(dev_data, data, length * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	scanKernel<<<blockCount, MAX_THREADS_PER_BLOCK>>>(dev_data, length, NULL);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "scan launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching scan!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(*out, dev_data, length * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	if (blockCount > 1) {

		int *remaining = (int*)malloc(sizeof(int)*blockCount);
		if (remaining == NULL) ERR("Failed to allocate memory");
		for (int i = 0; i < blockCount; i++) {
			if (i*MAX_THREADS_PER_BLOCK - 1 < length)
				remaining[i] = (*out)[(i+1) * MAX_THREADS_PER_BLOCK - 1] + data[(i+1) * MAX_THREADS_PER_BLOCK - 1];
			else remaining[i] = (*out)[(i+1)*MAX_THREADS_PER_BLOCK - 1] + data[length - 1];
		}
		int *scanned;
		cudaScan(remaining, blockCount, &scanned);

		int *dev_scanned;
		_cudaMalloc((void**)&dev_scanned, blockCount * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}
		cudaMemset((void*)dev_data, 0, blockCount *sizeof(int));

		cudaStatus = cudaMemcpy(dev_scanned, scanned, blockCount * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		addPerBlockKernel << <blockCount, MAX_THREADS_PER_BLOCK >> >(dev_data,dev_scanned);

		cudaStatus = cudaMemcpy(*out, dev_data, length * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		cudaFree(dev_scanned);

		free(scanned);
		free(remaining);
	}

Error:
	cudaFree(dev_data);
	return cudaStatus;
}


/******************************************
			MAIN FUNCTION
******************************************/
int main()
{
	test_scan(&cudaScan);
    return 0;
}