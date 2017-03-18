/******************************************
			INCLUDES
******************************************/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
/******************************************
			MACROS
******************************************/
#define ERR(s,...) (fprintf(stderr, s,__VA_ARGS__),\
						exit(EXIT_FAILURE))
/******************************************
			CONSTANTS
******************************************/
#define KB 1024
#define MB KB*1024
#define MAX_THREADS_PER_BLOCK 1024
#define MAX_BLOCKS_PER_GRID 65535

/******************************************
			FUNCTION DECLARATIONS
******************************************/
__global__ void scanKernel(int *data, const int len, int *out);

cudaError_t cudaScan(int *arr, const int length, int** out);

/* TESTS */
void test_scan();
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

/* HELPER FUNCTIONS */

/* KERNEL WRAPPING FUNCTIONS */
cudaError_t cudaScanR(int *data, const int length, int **out) {
	int *dev_data;
	cudaError_t cudaStatus;

	*out = (int*)malloc(sizeof(int)*length);
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

	scanKernel << <blockCount, MAX_THREADS_PER_BLOCK >> >(dev_data, length, NULL);

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
			if (i*MAX_THREADS_PER_BLOCK-1 < length)
				remaining[i] = *out[i * MAX_THREADS_PER_BLOCK - 1] + data[i * MAX_THREADS_PER_BLOCK - 1];
			else remaining[i] = *out[i*MAX_THREADS_PER_BLOCK - 1] + data[length - 1];
		}
		int *scanned;
		cudaScanR(remaining, blockCount, &scanned);

		for (int i = 0; i < blockCount; i++) {
			printf("%d ", scanned[i]);
		}
	}

Error:
	cudaFree(dev_data);
	return cudaStatus;
}

cudaError_t cudaScan(int *data, const int length, int **out) {
	int *dev_data;
	cudaError_t cudaStatus;

	int count = 0;
	for (int i = 0; i < length; i++) count++;
	printf("counted %d\n\n", count);

	*out = (int*)malloc(sizeof(int)*length);
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


		for (int i = 0; i < length; i++) {
			(*out)[i] += scanned[i/MAX_THREADS_PER_BLOCK];
		}


		free(scanned);
		free(remaining);
	}

Error:
	cudaFree(dev_data);
	return cudaStatus;
}


void test_scan() {
	int len = 1024;
	int n = 16;
	// n = 17 wont pass because limit of blocks will be exceeded (64 MB of data - max blocks is 65535*1024 < 64 MB)
	for (int i = 0; i < n; i++) {
		int *arr = (int*)malloc(sizeof(int)*len);
		if (arr == NULL) {
			printf("Failed to allocate memory\n");
			exit(1);
		}

		for (int i = 0; i < len; i++) {
			arr[i] = 1;
		}

		int *out;
		cudaScan(arr, len, &out);

		int l = out[len - 1];

		if (out[len - 1] != len-1) ERR("test_scan %d failed",i);
		free(arr);
		free(out);
		len = len << 1;
	}
}

/******************************************
			MAIN FUNCTION
******************************************/
int main()
{
	test_scan();
    return 0;
}