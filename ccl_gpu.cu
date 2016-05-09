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
#include "prefix_sum.cuh"

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

__device__ void add_vertex(int loc1, int loc2, int *d_vertex) {
	int d = ++d_vertex[loc1*9];
	d_vertex[loc1*9+d] = loc2;	
}


__global__ void get_vertex(int w, int h, int *d_vertex, unsigned char *img, unsigned char byF) {
	int loc = get_loc();
	int i;
	int x = get_x();
	int y = get_y();
	int xx,yy;
	if(loc>=w*h||img[loc]!=byF) return;
	for(i=0;i<8;i++) {
		xx = x + dx[i];
		yy = y + dy[i];
		if(check_bound(xx,yy,w,h)&&img[xx+yy*w]==byF) add_vertex(loc,xx+yy*w,d_vertex);
	}
}

__global__ void init_label(int w, int h, int *label) {
	int loc = get_loc();
	if(loc>=w*h) return;
	label[loc] = loc;
}

__global__ void CCL(int w, int h, int *d_label, int *d_vertex, bool *d_flag, int *d_flag_1, int *d_flag_2) {
	int loc = get_loc();
	int i,j;
	int col_i,col_j;

	if(loc>=w*h) return;

	int flag = 0;

	d_flag_1[loc] = 0;
	col_i = d_label[loc];

	for(i=1;i<=d_vertex[loc*9];i++) {
		j = d_vertex[loc*9+i];
		col_j = d_label[j];
		if(col_i == col_j) continue;
		if(col_i < col_j) {
			} else if(col_i > col_j) {
				col_i = col_j;
				flag = 1;
		}
	}
	if(flag) {
		d_label[loc] = col_i; 
		d_flag_2[loc] = 1;
		*d_flag = 1;
	}               
}

__global__ void get_label_num(int w, int h, unsigned char *img, int *d_label, unsigned char byF, int *d_lable_num) {
	int loc = get_loc();
	if(loc>=w*h) return;
	if(img[loc] == byF && d_label[loc] == loc) {
		atomicAdd(d_lable_num, 1);	
	}

}
__global__ void set_image(int w, int h, unsigned char *img, int *d_label, int *imgOut, unsigned char byF, int *d_lable_num) {
	int loc = get_loc();
	if(loc>=w*h) return;
	if (img[loc] == byF){
		INT_PTR(imgOut[loc]) = d_label[loc];
		if(d_label[loc] == loc) {
			printf("!! %d\n",loc);
			atomicAdd(d_lable_num, 1);
		}
	}
}



__global__ void init_flag_false(bool *d_flag) {
	*d_flag = 0;
}

__global__ void test(int *lable, int w, int h) {
	for(int i=0;i<h;i++)
		for(int j=0;j<w;j++) printf("%d %d %d\n",w,h,lable[j+i*w]);
}


int gpuLabelImage(int w, int h, int ws, int wd, unsigned char *img, int *imgOut, unsigned char byF,int *numLabels)
{

	
	bool flag = 0;
	const bool false_flag = 0;

	bool *d_flag;
	int *d_edge_num;
	int *d_label_num;
	int *d_label;
	int *d_vertex;
	int *d_flag_1;
	int *d_flag_2;

	const int MAX_VERTEX = w*h;

	cudaMalloc(&d_flag, sizeof(bool));
	cudaMalloc(&d_edge_num, sizeof(int));
	cudaMalloc(&d_label_num, sizeof(int));
	cudaMalloc(&d_label, sizeof(int)*MAX_VERTEX);
	cudaMalloc(&d_vertex, sizeof(int)*MAX_VERTEX*9);
	cudaMalloc(&d_flag_1, sizeof(int)*MAX_VERTEX);
	cudaMalloc(&d_flag_2, sizeof(int)*MAX_VERTEX);
	
	//cudaMemset(d_vertex, 0, sizeof(int)*MAX_VERTEX*9);
	cudaMemset(d_flag_1, 1, sizeof(int)*MAX_VERTEX);
	//cudaMemset(d_flag_2, 0, sizeof(int)*MAX_VERTEX);
	
	const dim3 get_vertex_blocks_rect(1,h);
	const dim3 get_vertex_threads_rect(w,1);
	//const dim3 get_vertex_blocks_rect(w/32+1,h/32+1);
	//const dim3 get_vertex_threads_rect(32,32);


	const dim3 init_label_blocks_rect(1,h);
	const dim3 init_label_threads_rect(w,1);
	//const dim3 init_label_blocks_rect(w/32+1,h/32+1);
	//const dim3 init_label_threads_rect(32,32);

	const dim3 ccl_blocks_rect(1,h);
	const dim3 ccl_threads_rect(w,1);
	//const dim3 ccl_blocks_rect(w/32+1,h/32+1);
	//const dim3 ccl_threads_rect(32,32);


	const dim3 set_image_blocks_rect(1,h);
	const dim3 set_image_threads_rect(w,1);
	//const dim3 set_image_blocks_rect(w/32+1,h/32+1);
	//const dim3 set_image_threads_rect(32,32);


	const dim3 get_label_num_blocks_rect(1,h);
	const dim3 get_label_num_threads_rect(w,1);
	//const dim3 get_label_num_blocks_rect(w/32+1,h/32+1);
	//const dim3 get_label_num_threads_rect(32,32);
	
	
	get_vertex<<<get_vertex_blocks_rect, get_vertex_threads_rect>>>(w,h,d_vertex,img,byF);
	
	init_label<<<init_label_blocks_rect, init_label_threads_rect>>>(w,h,d_label);

	do {
		init_flag_false<<<1,1>>>(d_flag);
		CCL<<<ccl_blocks_rect, ccl_threads_rect>>>(w,h,d_label,d_vertex,d_flag,d_flag_1,d_flag_2);
		
		std::swap(d_flag_1,d_flag_2);

		CUDA_CALL(cudaMemcpy(&flag, d_flag, sizeof(bool), cudaMemcpyDeviceToHost));
	} while(flag == 1);

	//get_label_num<<<get_label_num_blocks_rect, get_label_num_threads_rect>>>(w,h,img,d_label,byF,d_label_num);
	set_image<<<set_image_blocks_rect, set_image_threads_rect>>>(w,h,img,d_label,imgOut,byF,d_label_num);
	cudaMemcpy(numLabels, d_label_num, sizeof(int), cudaMemcpyDeviceToHost);

	return 0;

}
 