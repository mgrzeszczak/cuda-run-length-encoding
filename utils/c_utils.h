#pragma once

#include "consts.h"

void* _malloc(int size) {
	void* mem = malloc(size);
	if (mem == NULL) ERR("Failed to malloc");
	return mem;
}

void* _realloc(void *data, int size) {	
	void* mem = realloc(data,size);
	if (mem == NULL) ERR("Failed to realloc");
	return mem;
}