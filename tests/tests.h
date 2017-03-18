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

void test_mask() {
	int len = 1024;
	int *data = (int*)malloc(sizeof(int)*len);
	for (int i = 0; i < len; i++) data[i] = i;
}
