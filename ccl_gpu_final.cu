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
#include "common.cu"


__global__ void init_label(int w, int h, int *d_label) {
	int loc = get_loc();
	if(loc>=w*h) return;
	d_label[loc] = loc;
}

__global__ void init_flag(bool *d_flag) {
	*d_flag = 0;
}

__device__ int find(int loc, int *d_label) {
	int temp = loc;
	while(temp != d_label[temp]) temp = d_label[temp];
	int ans = temp;
	temp = loc;
	while(temp != ans) {
		temp = d_label[temp];
		d_label[temp] = ans;
	}
	return ans;
}

__global__ void ccl_find(int w, int h, unsigned char *img, unsigned char byF, int *d_label, int *d_flag) {
	int loc = get_loc();
	int x = get_x();
	int y = get_y();

	if(loc>=w*h) return;

	int min_label = w*h+1;

	for(int i=0;i<8;i++) {
		xx = x + dx[i];
		yy = y + dy[i];
		if(check_connect(xx+yy*w)) {
			min_label = min(min_label, find(xx+yy*w, d_label));
		}
	}

	//__syncthreads();

	if(min_label<d_label[loc]) {
		d_label[loc] = min_label;
		*d_flag = 1;
	}
}

__global__ void set_image(int w, int h, unsigned char *img, unsigned char byF, int *imgOut) {
	int loc = get_loc();
	if(loc>=w*h||img[loc]!=byF) return;
	INT_PTR(imgOut[loc]) = find(loc);
}

int gpuLabelImage(int w, int h, int ws, int wd, unsigned char *img, int *imgOut, unsigned char byF,int *numLabels)
{

	bool flag = 0;


	int *d_label;
	int *d_fa;
	int *d_flag;



	cudaMalloc(&d_label, sizeof(int)*w*h);
	cudaMalloc(&d_fa, sizeof(int)*w*h);
	cudaMalloc(&d_flag, sizeof(bool));

	const threads_pre_blocks = 256; 
	int blocks_pre_row = w/;
	if(w%threads_pre_blocks) blocks_pre_row++
	const dim3 init_flag_blocks_rect(blocks_pre_row,h,1);
	const dim3 init_flag_threads_rect(threads_pre_blocks,1,1);

	const dim3 ccl_find_blocks_rect(blocks_pre_row,h,1);
	const dim3 ccl_find_threads_rect(threads_pre_blocks,1,1);

	const dim3 set_image_blocks_rect(blocks_pre_row,h,1);
	const dim3 set_image_threads_rect(threads_pre_blocks,1,1);

	init_label<<<init_flag_blocks_rect,init_flag_threads_rect>>>(w,h,d_label);

	do {
		init_flag<<<1,1>>>(d_flag);
		ccl_find<<<ccl_find_blocks_rect,ccl_find_threads_rect>>>(w,h,img,byF,d_label,d_flag);
		cudaMemcpy(&flag,d_flag,sizeof(bool),cudaMemcpyDeviceToHost);
		}while(flag)
	}
	set_image<<<>>>(w,h,img,byf,imgOut);

	return 0;
	
	
}
