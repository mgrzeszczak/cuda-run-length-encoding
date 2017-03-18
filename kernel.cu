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

/******************************************
			MAIN FUNCTION
******************************************/
int main()
{
	test_scan(&cudaScan);
    return 0;
}