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

using namespace std;

__device__ int dx[8] = {-1,0,1,-1,1,-1,0,1};
__device__ int dy[8] = {-1,-1,-1,0,0,1,1,1};


__device__ int get_loc(int w, int h) {
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if(idx>=w||idy>=h) {
		//printf("yes\n");
		return w*h+1;
	} else {
		int loc = w * idy +idx;
	return loc;
	}
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

__device__ bool check_connect(int loc1, int loc2, unsigned char *img, unsigned char byF) {
	if(img[loc1]==byF&&img[loc2]==byF) return 1;
		else return 0;
}

__global__ void init_label(int w, int h, int *d_label, int *d_fa) {
	int loc = get_loc(w,h);
	if(loc>=w*h) return;
	d_label[loc] = d_fa[loc] = loc;
}

__global__ void init_flag(bool *d_flag) {
	*d_flag = 0;
}

__device__ int find(int loc, int *d_label, bool *d_flag) {
	int temp = loc;
	if(temp==d_label[temp]) return temp;
	while(temp != d_label[temp]) temp = d_label[temp];
	int ans = temp;
	temp = loc;
	while(temp != ans) {
		temp = d_label[temp];
		d_label[temp] = ans;
	}
	*d_flag = 1;
	return ans;
}

__global__ void ccl_find(int w, int h, unsigned char *img, unsigned char byF, int *d_label, int *d_fa, bool *d_flag) {
	int loc = get_loc(w,h);
	int x = get_x();
	int y = get_y();
	int xx,yy;

	if(loc>=w*h||img[loc]!=byF) return;

	int min_label = w*h+1;

	for(int i=0;i<8;i++) {
		xx = x + dx[i];
		yy = y + dy[i];
		if(check_bound(xx,yy,w,h)&&img[xx+yy*w]==byF) {
			//int temp = find(xx+yy*w,d_label,d_flag);
			int temp = d_label[xx+yy*w];
			//printf("x %d y %d temp %d min_temp %d\n",xx,yy,temp,min_label);
			if(temp<min_label) min_label = temp;
			//min_label = min(min_label, find(xx+yy*w, d_label));
		}
	}

	//__syncthreads();
	//printf("min %d loc %d\n",min_label,d_label[loc]);
	if(min_label<d_label[loc]) {
		d_fa[d_label[loc]] = min_label;
		*d_flag = 1;
		//printf("min %d loc %d\n",min_label,d_label[loc]);
	}
}

__global__ void ccl_merge(int w,int h, int *d_label, int *d_fa) {
	int loc = get_loc(w,h);
	if(loc>=w*h) return;
	int label = d_label[loc];
	int temp;
	if(label == loc) {
		do {
			temp = label;
			label = d_fa[label];
		}while(temp!=label);
		d_fa[loc] = label;
	}
}

__global__ void ccl_update(int w, int h, int *d_label, int *d_fa) {
	int loc =get_loc(w,h);
	if(loc>=w*h) return;
	d_label[loc] = d_fa[d_fa[d_label[loc]]];

}

__global__ void get_label_num(int w, int h, unsigned char *img, int *d_label, unsigned char byF, int *d_lable_num) {
	int loc = get_loc(w,h);
	if(loc>=w*h) return;
	if(img[loc] == byF && d_label[loc] == loc) {
		//printf("%d\n",loc);
		atomicAdd(d_lable_num, 1);
	}
}

__global__ void set_image(int w, int h, int *d_label, unsigned char *img, unsigned char byF, int *imgOut) {
	int loc = get_loc(w,h);
	if(loc>=w*h||img[loc]!=byF) return;
	//INT_PTR(imgOut[loc]) = find(loc,d_label,d_flag);
	int temp = loc;
	while(d_label[temp]!=temp) temp = d_label[temp];
	INT_PTR(imgOut[loc]) = temp;
	//printf("%d %d %d\n",loc,find(loc,d_label,d_flag),d_label[loc]);
}

int gpuLabelImage(int w, int h, int ws, int wd, unsigned char *img, int *imgOut, unsigned char byF,int *numLabels)
{

	bool flag = 0;

	int *d_label;
	int *d_fa;
	bool *d_flag;
	int *d_label_num;

	cudaMalloc(&d_label, sizeof(int)*w*h);
	cudaMalloc(&d_fa, sizeof(int)*w*h);
	cudaMalloc(&d_flag, sizeof(bool));
	cudaMalloc(&d_label_num, sizeof(int));
	
	const int threads_pre_blocks = 256; 
	int blocks_pre_row = w/threads_pre_blocks;
	if(w%threads_pre_blocks) blocks_pre_row++;
	
	const dim3 init_flag_blocks_rect(blocks_pre_row,h,1);
	const dim3 init_flag_threads_rect(threads_pre_blocks,1,1);

	const dim3 ccl_find_blocks_rect(blocks_pre_row,h,1);
	const dim3 ccl_find_threads_rect(threads_pre_blocks,1,1);

	const dim3 ccl_merge_blocks_rect(blocks_pre_row,h,1);
	const dim3 ccl_merge_threads_rect(threads_pre_blocks,1,1);

	const dim3 ccl_update_blocks_rect(blocks_pre_row,h,1);
	const dim3 ccl_update_threads_rect(threads_pre_blocks,1,1);

	const dim3 set_image_blocks_rect(blocks_pre_row,h,1);
	const dim3 set_image_threads_rect(threads_pre_blocks,1,1);

	const dim3 get_label_num_blocks_rect(blocks_pre_row,h,1);
	const dim3 get_label_num_threads_rect(threads_pre_blocks,1,1);
	
	/*
	const dim3 init_flag_blocks_rect(1,h,1);
	const dim3 init_flag_threads_rect(w,1,1);

	const dim3 ccl_find_blocks_rect(1,h,1);
	const dim3 ccl_find_threads_rect(w,1,1);

	const dim3 ccl_merge_blocks_rect(1,h,1);
	const dim3 ccl_merge_threads_rect(w,1,1);

	const dim3 ccl_update_blocks_rect(1,h,1);
	const dim3 ccl_update_threads_rect(w,1,1);

	const dim3 set_image_blocks_rect(1,h,1);
	const dim3 set_image_threads_rect(w,1,1);

	const dim3 get_label_num_blocks_rect(1,h,1);
	const dim3 get_label_num_threads_rect(w,1,1);
	*/

	init_label<<<init_flag_blocks_rect,init_flag_threads_rect>>>(w,h,d_label,d_fa);

	do {
		init_flag<<<1,1>>>(d_flag);
		ccl_find<<<ccl_find_blocks_rect,ccl_find_threads_rect>>>(w,h,img,byF,d_label,d_fa,d_flag);
		cudaMemcpy(&flag,d_flag,sizeof(bool),cudaMemcpyDeviceToHost);

		if(flag) {
			ccl_merge<<<ccl_merge_blocks_rect, ccl_merge_threads_rect>>>(w,h,d_label,d_fa);
			ccl_update<<<ccl_update_blocks_rect, ccl_update_threads_rect>>>(w,h,d_label, d_fa);
		} else break;
		//printf("running\n");

	}while(flag);
	
	get_label_num<<<get_label_num_blocks_rect,get_label_num_threads_rect>>>(w,h,img,d_label,byF,d_label_num);
	cudaMemcpy(numLabels, d_label_num, sizeof(int), cudaMemcpyDeviceToHost);

	set_image<<<set_image_blocks_rect, set_image_threads_rect>>>(w,h,d_fa,img,byF,imgOut);


	return 0;
	
	
}
