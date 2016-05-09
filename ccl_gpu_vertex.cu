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
#include <cstdlib>

#include "ccl_gpu.cuh"

#define CUDA_CALL(x) {const cudaError_t a = (x);if (a != cudaSuccess) { printf("\nCuda error: %s (err_num = %d)\n",cudaGetErrorString(a),a);cudaDeviceReset();}}
#define MAX_HEIGHT 1024
#define MAX_WIDTH 1024
#define MAX_VERTEX MAX_WIDTH * MAX_HEIGHT

#define INT_PTR(x) (*((int*)(&(x))))




__global__ void get_vertex() {
	

}

__global__ void init_label() {

}

__global__ void CCL() {

}

__global__ void get_label_num() {

}

__global__ void set_image() {

}

int gpuLabelImage(int w, int h, int ws, int wd, unsigned char *img, int *imgOut, unsigned char byF,int *numLabels)
{
	
	bool flag = 0;
	int edge_num = 0;
	const bool false_flag = 0;

	bool *d_flag;
	int *d_edge_num;
	int *d_lable_num;
	int *d_label;
	int *d_vertex;

	cudaMalloc(&d_flag, sizeof(bool));
	cudaMalloc(&d_edge_num, sizeof(int));
	cudaMalloc(&d_lable_num, sizeof(int));
	cudaMalloc(&d_label, sizeof(int)*MAX_VERTEX);
	cudaMalloc(&d_vertex, sizeof(int)*MAX_VERTEX*9);

	cudaMemset(&d_vertex, 0, sizeof(int)*MAX_VERTEX*9);
	
	get_vertex<<<>>>();
	init_label<<<>>();

	do {
		CCL<<<>>();

	} while(flag == 1);

	get_label_num<<<>>();

	set_image<<<>>>();


	return 0;

}
 