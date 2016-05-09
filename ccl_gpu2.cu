/*************************************************************************************
LIBRARY: Connected-Component Labeling (CCL)
FILE:    main.cu
DATE:    2/11/2014
UPDATED:  1 gpu parallel 2 edge set

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
#define MAX_EDGE MAX_HEIGHT * MAX_WIDTH * 8

#define INT_PTR(x) (*((int*)(&(x))))


__device__ int edgeSet[MAX_EDGE][2];
__device__ int dx[8] = {-1,0,1,-1,1,-1,0,1};
__device__ int dy[8] = {-1,-1,-1,0,0,1,1,1};
__device__ int count = 0;
__device__ int min_label = 9999999;

__device__ bool checkBound(int x, int y, int w, int h) {
	if (x>0&&y>0&&x<w&&y<h) return 1;
		else return 0;
}

__device__ void addEdge(int a, int b,int *edgeNum) {
	*edgeNum = *edgeNum + 1;
	edgeSet[*edgeNum][0] = a;
	edgeSet[*edgeNum][1] = b;
}



__global__ void getEdge(int w,int h,int ws,unsigned char *img, unsigned char byF, int *edgeNum) {
	int i;
	int x,y,xx,yy;
	for(y=0;y<h;y++)
		for(x=0;x<w;x++) {
			if (img[x + y*ws] != byF) continue;
			for(i = 0;i<8;i++) {
				xx = x + dx[i];
				yy = y + dy[i];
				if (checkBound(xx,yy,w,h)&&img[xx + yy*ws] == byF) addEdge(x + y * w, xx + yy * w, edgeNum);
		}
	}

}

__global__ void initLabel(int w, int h,int *label) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int loc = gridDim.x * blockDim.x * idy +idx;
	//if(loc == w*h-1) printf("initLabel\n");
	if(loc<w*h) {
		label[loc] = loc;
	}
}

 __global__ void CCL(int * label, bool * flag, int edgeNum) {
 	 int idx = blockIdx.x * blockDim.x + threadIdx.x;
	 int idy = blockIdx.y * blockDim.y + threadIdx.y;
	 int loc = gridDim.x * blockDim.x * idy +idx;
	 if(loc>=edgeNum) return;
	 int x = edgeSet[loc][0];
	 int y = edgeSet[loc][1];
	 if (label[x] == label[y]) return;
	 if (label[x] < label[y]) {
		 return;
	 } else {
		 label[x] = label[y];
		 *flag = 1;
		}
}	


 __global__ void getLabelNum(int w,int h, unsigned char *img, unsigned char byF, int *label, int *num_labels) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int loc = gridDim.x * blockDim.x * idy +idx;
	//if(loc==w*h-1) printf("get label num\n");
	if(loc>=w*h) return;
	if (label[loc] == loc && img[loc] == byF){
			atomicAdd(num_labels,1);
		 }
}

 __global__ void  setImg(int w, int h, int ws, int wd, unsigned char *img, unsigned char byF, int *label, int *imgOut) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int loc = gridDim.x * blockDim.x * idy +idx;
	//if(loc == w*h-1) printf("setimg\n");
	if(loc>=w*h) return;
	if (img[loc] == byF){
		//printf("col %d %d\n",loc,imgOut[loc]);
		INT_PTR(imgOut[loc]) = label[label[loc]];
		
	}
}


__global__ void resortLabel(int w, int h, int *label, unsigned char *img, unsigned char byF) {
 	int x,y;
 	for(y=0;y<h;y++)
 		for(x=0;x<w;x++){
 			if(label[x+y*w]==x+y*w&&img[x+y*w]==byF) {
 				atomicAdd(&count, 1);
 				label[x+y*w] = count;
				//printf("x %d y %d %d %d\n",x,y,x+y*w,label[x+y*w]);
 			}
		}
 }

 __global__ void testOnCuda(int w, int h, int ws, int wd , unsigned char *img, int *imgOut, unsigned char byF, int *label, int edge_num) {
	/*
	int x,y;
	int loc;
	int c= 0;
	for(x=0;x<h;x++)
		for(y=0;y<w;y++) {
			loc = x*w+y;
			if(img[loc] == byF&&label[loc]==loc) {
				printf("x= %d y = %d i %d %d %d\n",y,x,++c, loc,label[loc]);				
		}
	}
	*/
	 //printf("%d\n",edge_num);
 }

int gpuLabelImage(int w, int h, int ws, int wd, unsigned char *img, int *imgOut, unsigned char byF,int *numLabels)
{
	bool flag = 0;
	int edge_num = 0;
	const bool false_flag = 0;

	bool * d_flag = 0;
	int * d_edge_num = 0;
	int *label;
	int *d_label_num;

	const dim3 initLabel_blocks_rect(w/32+1, h/32+1);
	const dim3 initLabel_threads_rect(32,32);

	const dim3 getEdge_blocks_rect(1);
	const dim3 getEdge_threads_rect(1);
	
	const dim3 getLabelNum_blocks_rect(w/32+1,h/32+1);
	const dim3 getLabelNum_threads_rect(32,32);

	const dim3 resortLabel_blocks_rect(1);
	const dim3 resortLabel_threads_rect(1);

	const dim3 setImg_blocks_rect(w/32+1,h/32+1);
	const dim3 setImg_threads_rect(32,32);


	cudaMalloc(&d_flag, sizeof(bool));
	cudaMalloc(&d_edge_num, sizeof(int));
	cudaMalloc(&d_label_num, sizeof(int));
	cudaMalloc((void **) &label, sizeof(int)*MAX_HEIGHT*MAX_WIDTH);


	printf("w %d h %d ws %d wd %d byF %d\n",w,h,ws,wd,byF);

	getEdge<<<getEdge_blocks_rect, getEdge_threads_rect>>>(w,h,ws,img,byF,d_edge_num);
	
	
	initLabel<<<initLabel_blocks_rect, initLabel_threads_rect>>>(w, h, label);

	CUDA_CALL(cudaMemcpy(&edge_num, d_edge_num, sizeof(int), cudaMemcpyDeviceToHost));
	testOnCuda<<<1,1>>>(w,h,ws,wd,img,imgOut,byF,label,edge_num);

	//printf("edge num = %d\n",edge_num);

	const dim3 ccl_blocks_rect(16,16);
	const dim3 ccl_threads_rect(edge_num/256+1,1);

	
	do {

		CUDA_CALL(cudaMemcpy(d_flag, &false_flag , sizeof(bool), cudaMemcpyHostToDevice));
			
		//CCL2<<<1, 1>>>(label, d_flag, edge_num, h, w);

		CCL<<<ccl_blocks_rect, ccl_threads_rect>>>(label, d_flag, edge_num);
		
		CUDA_CALL(cudaMemcpy(&flag, d_flag, sizeof(bool), cudaMemcpyDeviceToHost));

		
	} while (flag == 1);

	
	getLabelNum<<<getLabelNum_blocks_rect, getLabelNum_threads_rect>>>(w,h,img,byF,label,d_label_num);
	CUDA_CALL(cudaMemcpy(numLabels, d_label_num, sizeof(int), cudaMemcpyDeviceToHost));


	resortLabel<<<resortLabel_blocks_rect, resortLabel_threads_rect>>>(w,h,label,img,byF);

	setImg<<<setImg_blocks_rect, setImg_threads_rect>>>(w,h,ws,wd,img,byF,label,imgOut);

	//testOnCuda<<<1,1>>>(w,h,ws,wd,img,imgOut,byF,label,edge_num);

	return 0;
}
 