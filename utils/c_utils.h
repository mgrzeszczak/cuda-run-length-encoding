#pragma once

#include "consts.h"

void* _malloc(int size) {
	void* mem = malloc(size);
	if (mem == NULL) ERR("Failed to malloc");
	return mem;
}