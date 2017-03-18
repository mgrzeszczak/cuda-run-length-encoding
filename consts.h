#pragma once
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
