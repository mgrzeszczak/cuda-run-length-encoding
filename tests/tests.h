#pragma once
#include "../utils/consts.h"
#include "../utils/c_utils.h"
#include <stdarg.h>

void test_scan(void(*cudaScan)(int *, const int , int **, bool)) {
	int len = 1024;
	int n = 16;
	// n = 17 wont pass because limit of blocks will be exceeded (64 MB of data - max blocks is 65535*1024 < 64 MB)
	for (int i = 0; i < n; i++) {
		int *arr = (int*)_malloc(sizeof(int)*len);

		for (int i = 0; i < len; i++) {
			arr[i] = 1;
		}

		int *out;
		cudaScan(arr, len, &out,true);

		int l = out[len - 1];

		if (out[len - 1] != len ) printf("test_scan %d failed\n", i);
		else printf("test_scan [%d] - SUCCESS\n",i+1);
		free(arr);
		free(out);
		len = len << 1;
	}
}

void test_mask(void(*cudaMask)(char *, const int, int **)) {
	srand(123);

	int len = 1024;
	int max = 10;
	
	int n = 16;
	int m = 1;
	for (int i = 0; i < n; i++) {

		for (int j = 0; j < m; j++) {
			char *data = (char*)malloc(sizeof(char)*len);

			int expected = len;
			for (int i = 0; i < len; i++) {
				data[i] = (char)(rand() % max);
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

typedef struct compressed_mask_test_case {
	int* data;
	int length;
	int* compressed;
	int compressedLength;
} cm_test;

cm_test* create_cm_test_case(int *data, int length, int* compressed, int compressedLength) {
	cm_test* test = (cm_test*)malloc(sizeof(cm_test));
	test->data = data;
	test->length = length;
	test->compressed = compressed;
	test->compressedLength = compressedLength;
	return test;
}

void free_cm_test_case(cm_test* test) {
	free(test->data);
	free(test->compressed);
	free(test);
}

int* create_array(int length, ...) {
	va_list args;
	va_start(args, length);

	int *arr = (int*)_malloc(sizeof(int)*length);

	for (int i = 0; i < length; i++) {
		int val = va_arg(args, int);
		arr[i] = val;
	}

	va_end(args);
	return arr;
}

void single_test_compressed_mask(int no, cm_test* test, void(*cudaCompressedMask)(int *mask, const int length, int **compressedMask, int *compressedLength)) {
	int *compressed;
	int compressedLength;
	cudaCompressedMask(test->data, test->length, &compressed,&compressedLength);

	int failed = 0;
	if (compressedLength != test->compressedLength) {
		printf("test_compressed_mask [%d] FAILED - invalid length, expected = %d, got = %d\n", no,test->compressedLength,compressedLength);
		failed = 1;
	}
	if (failed == 0) {

		for (int i = 0; i < compressedLength; i++) {
			if (compressed[i] != test->compressed[i]) {
				printf("test_compressed_mask [%d] FAILED - different output\n", no);
				failed = 1;
				break;
			}
		}

		if (failed == 1) {
			printf("Expected:\n");
			for (int i = 0; i < compressedLength; i++) {
				printf("%d ", test->compressed[i]);
			}
			printf("\n");
			printf("Got:\n");
			for (int i = 0; i < compressedLength; i++) {
				printf("%d ", compressed[i]);
			}
			printf("\n");
		}
	}

	if (failed == 0) {
		printf("test_compressed_mask [%d] SUCCESS\n", no);
	}

	free(compressed);
	free_cm_test_case(test);
}

void test_compressed_mask(void(*cudaCompressedMask)(int *mask, const int length, int **compressedMask, int *compressedLength)) {
	single_test_compressed_mask(1, 
		create_cm_test_case(create_array(8,1,2,3,4,4,4,5,5),	8,
							create_array(6,0,1,2,3,6,8),		6), 
		cudaCompressedMask);
	single_test_compressed_mask(2,
		create_cm_test_case(create_array(10, 1,1,1,1,1,1,1,1,1,1), 10,
			create_array(2, 0, 10), 2),
		cudaCompressedMask);

	single_test_compressed_mask(3,
		create_cm_test_case(create_array(20, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,2,2,2,2,2,3,3,3,3,3), 20,
			create_array(4, 0,10,15,20), 4),
		cudaCompressedMask);
}

void test_counts(void(*cudaCounts)(int *compressedMask, int length, int **counts)) {
	int len = 10;
	int *mask = (int*)_malloc(sizeof(int) * len);

	for (int i = 0; i < len; i++) {
		mask[i] = i << len;
	}

	int *counts;
	cudaCounts(mask, len, &counts);

	int failed = 0;
	for (int i = 0; i < len - 1; i++) {
		if (counts[i] != mask[i + 1] - mask[i]) {
			printf("test_counts FAILED\n");
			failed = 1;
			break;
		}
	}

	if (failed == 0) {
		printf("test_counts SUCCESS\n");
	}
	free(mask);
	free(counts);
}

void test_symbols(void (*cudaSymbols)(int *compressedMask, int maskLength, char *data, int dataLength, char **symbols)) {
	char data[] = { 'a','a','a','b','b','b,','c','c','d','e' };

	int mask[] = {0,3,6,8,9,10};
	int maskLength = 6;
	int dataLength = 10;

	char *symbols;
	cudaSymbols(mask, maskLength, data, dataLength, &symbols);

	for (int i = 0; i < maskLength - 1; i++) {
		printf("count = %d, symbol = %c\n", mask[i + 1] - mask[i], symbols[i]);
	}

	free(symbols);
}
