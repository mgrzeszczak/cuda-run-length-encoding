#pragma once
#include "../utils/consts.h"
#include "../utils/c_utils.h"

void test_scan(void(*cudaScan)(int *, const int , int **)) {
	int len = 1024;
	int n = 16;
	// n = 17 wont pass because limit of blocks will be exceeded (64 MB of data - max blocks is 65535*1024 < 64 MB)
	for (int i = 0; i < n; i++) {
		int *arr = (int*)_malloc(sizeof(int)*len);

		for (int i = 0; i < len; i++) {
			arr[i] = 1;
		}

		int *out;
		cudaScan(arr, len, &out);

		int l = out[len - 1];

		if (out[len - 1] != len - 1) ERR("test_scan %d failed", i);
		else printf("test_scan [%d] - SUCCESS\n",i+1);
		free(arr);
		free(out);
		len = len << 1;
	}
}

void test_mask(void(*cudaMask)(int *, const int, int **)) {
	srand(123);

	int len = 1024;
	int max = 10;
	
	int n = 16;
	int m = 3;
	for (int i = 0; i < n; i++) {

		for (int j = 0; j < m; j++) {
			int *data = (int*)malloc(sizeof(int)*len);

			int expected = len;
			for (int i = 0; i < len; i++) {
				data[i] = rand() % max;
				if (i > 0 && data[i - 1] == data[i]) expected--;
			}

			int *mask;
			cudaMask(data, len, &mask);

			int sum = 0;
			for (int i = 0; i < len; i++) {
				sum += mask[i];
			}
			
			if (sum != expected) ERR("test_scan %d - FAILED\n", m*i+j+1);
			else printf("test_mask [%d] - SUCCESS\n", m*i+j+1);

			free(data);
			free(mask);
		}
		len = len << 1;
	}
}
