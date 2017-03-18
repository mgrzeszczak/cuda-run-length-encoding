/******************************************
			INCLUDES
******************************************/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>

#include "utils/cuda_utils.h"
#include "utils/c_utils.h"
#include "utils/consts.h"

#include "tests/tests.h"

#include "kernels/scan.h"
#include "kernels/mask.h"
#include "kernels/compressed_mask.h"
#include "kernels/counts.h"
#include "kernels/symbols.h"

/******************************************
			MAIN FUNCTION
******************************************/
void run_tests() {
	test_scan(&cudaScan);
	test_mask(&cudaMask);
	test_compressed_mask(&cudaCompressedMask);
	test_counts(&cudaCounts);
	test_symbols(&cudaSymbols);
}

int main()
{
	run_tests();
    return 0;
}