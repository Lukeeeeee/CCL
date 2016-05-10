/*************************************************************************************
LIBRARY: Connected-Component Labeling (CCL)
FILE:    main.cu
DATE:    2/11/2014
UPDATED: 

Contains the interface of CCL CPU alg. Finish your own gpu CCL alg in this file.
**************************************************************************************/

/**********************************************************************************************
***********************************************************************************************
#cat: gpuLabelImage - CCL GPU alg                       

Input:
w           - width of the image in pixels ÊäÈëÍ¼ÏñµÄÏñËØ¿í¶È X
h           - height of the image in pixels ÊäÈëÍ¼ÏñµÄÏñËØ¸ß¶È Y
ws          - pitch of the source image in bytes
wd          - pitch of the destination image in bytes 
img         - source image ÊäÈëÍ¼Ïñ
byF         - foreground mark (always 1 in this driver) Í¼ÏñµÄ±êÊ¶·û

Output:
numLabels   - The number of Labels (targets) in the image ÎïÌåµÄ¸öÊý
imgOut      - destination image Êä³öµÄÍ¼Ïñ

Return Codes:
reserved
**********************************************************************************************/
#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <ctime>
#include <iostream>
#include <cstdlib>
#include <algorithm>

#include "ccl_gpu.cuh"

#define CUDA_CALL(x) {const cudaError_t a = (x);if (a != cudaSuccess) { printf("\nCuda error: %s (err_num = %d)\n",cudaGetErrorString(a),a);cudaDeviceReset();}}
#define MAX_HEIGHT 1024
#define MAX_WIDTH 1024
//#define MAX_VERTEX MAX_WIDTH * MAX_HEIGHT

#define INT_PTR(x) (*((int*)(&(x))))

__device__ int dx[8] = {-1,0,1,-1,1,-1,0,1};
__device__ int dy[8] = {-1,-1,-1,0,0,1,1,1};

__device__ int get_loc() {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	return gridDim.x * blockDim.x * idy +idx;
}

__device__ int get_x() {
	return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ int get_y() {
	return blockIdx.y * blockDim.y + threadIdx.y;
}

__device__ bool check_bound(int x, int y, int w, int h) {
	if (x>0&&y>0&&x<w&&y<h) return 1;
		else return 0;
}


int gpuLabelImage(int w, int h, int ws, int wd, unsigned char *img, int *imgOut, unsigned char byF,int *numLabels)
{


	return 0;
	
	
}
