/*************************************************************************************
LIBRARY: Connected-Component Labeling (CCL)
FILE:    main.cu
DATE:    2/11/2014
UPDATED:  1 gpu serial 2 edge set

Contains the interface of CCL CPU alg. Finish your own gpu CCL alg in this file.
**************************************************************************************/

/**********************************************************************************************
***********************************************************************************************
#cat: gpuLabelImage - CCL GPU alg                       

Input:
w           - width of the image in pixels 输入图像的像素宽度 X
h           - height of the image in pixels 输入图像的像素高度 Y
ws          - pitch of the source image in bytes
wd          - pitch of the destination image in bytes 
img         - source image 输入图像
byF         - foreground mark (always 1 in this driver) 图像的标识符

Output:
numLabels   - The number of Labels (targets) in the image 物体的个数
imgOut      - destination image 输出的图像

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



int col[MAX_HEIGHT * MAX_WIDTH];

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
	//atomicAdd(edgeNum, 1);
	*edgeNum = *edgeNum + 1;
	edgeSet[*edgeNum][0] = a;
	edgeSet[*edgeNum][1] = b;
	//printf("edge %d %d %d\n",*edgeNum,a,b);
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
	int i,loc;
	for(i=0;i<w;i++) {
		loc = i + w * threadIdx.x;
		label[loc] = loc;
	}
}

 __global__ void CCL(int * label, bool * flag, int edgeNum) {
	 int id = threadIdx.x + blockIdx.x * gridDim.x;
	 if (id >= edgeNum) return;
	 int x = edgeSet[id][0];
	 int y = edgeSet[id][1];
	 int labelX = label[x];
	 int labelY = label[y];
	 if (labelX == labelY) return ;
	 if (labelX < labelY) {
		 atomicAdd(&label[y], labelX - labelY);
		 *flag = 1 ;
		} else {
			atomicAdd(&label[x], labelY - labelX);
			*flag = 1;
		}	

 }
 __global__ void CCL2(int * label, bool * flag, int edgeNum,int h,int w) {
	 int id;
	 for(id=0;id<edgeNum;id++) {
	 int x = edgeSet[id][0];
	 int y = edgeSet[id][1];
	 int labelX = label[x];
	 int labelY = label[y];
	 if (labelX == labelY) continue;
	 if (labelX < labelY) {
		 label[y] = label[x];
		 *flag = 1 ;
		} else {
			label[x] = label[y];
			*flag = 1;
		}	
	 }
 }

 __global__ void getLabelNum(int w,int h, unsigned char *img, unsigned char byF, int *label, int *num_labels) {
	 int y = threadIdx.x;
	 int x;
	 for(x=0;x<w;x++) {
		 if (label[x+y*w] == x+y*w && img[x+y*w] == byF){
			atomicAdd(num_labels,1);
		 }
	 }
 }


 __global__ void  setImg(int w, int h, int ws, int wd, unsigned char *img, unsigned char byF, int *label, int *imgOut) {
	 int x;
	 int y = threadIdx.x;
	 for(x=0;x<w;x++) {
		 if (img[x+y*ws] == byF){
			INT_PTR(imgOut[x+y*w]) = label[label[x+y*ws]];
			//label[x+y*ws]%=1000;
			//printf("get %d %d %d\n",x+y*ws, imgOut[x+y*wd], label[x+y*ws]);
		 }
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
	int x,y;
	int loc;
	int c= 0;
	for(x=0;x<h;x++)
		for(y=0;y<w;y++) {
			loc = x*w+y;
			if(img[loc] == byF&&label[loc]==loc) {
				label[loc] %=1000;
				printf("x= %d y = %d i %d %d %d\n",y,x,++c, loc,label[loc]);
				
		}
	}
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
	//int *label_index;

	unsigned char * img_local = (unsigned char *)malloc(w*h*sizeof(unsigned char));
	

	cudaMalloc(&d_flag, sizeof(bool));
	cudaMalloc(&d_edge_num, sizeof(int));
	cudaMalloc(&d_label_num, sizeof(int));
	cudaMalloc((void **) &label, sizeof(int)*MAX_HEIGHT*MAX_WIDTH);


	printf("w %d h %d ws %d wd %d byF %d\n",w,h,ws,wd,byF);

	getEdge<<<1,1>>>(w,h,ws,img,byF,d_edge_num);
	initLabel<<<1,h>>>(w, h, label);
	

	CUDA_CALL(cudaMemcpy(&edge_num, d_edge_num, sizeof(int), cudaMemcpyDeviceToHost));
	printf("edge num = %d\n",edge_num);

	//testOnCuda<<<1,1>>>(w,h,edge_num);
	
	CUDA_CALL(cudaMemcpy(col, label , sizeof(int)*MAX_HEIGHT * MAX_WIDTH, cudaMemcpyDeviceToHost));
	
	do {

		CUDA_CALL(cudaMemcpy(d_flag, &false_flag , sizeof(bool), cudaMemcpyHostToDevice));
			
		//CCL<<<block, threadPerBlock>>>(label, d_flag, edge_num);
		CCL2<<<1, 1>>>(label, d_flag, edge_num, h, w);
		
		CUDA_CALL(cudaMemcpy(&flag, d_flag, sizeof(bool), cudaMemcpyDeviceToHost));

		
	} while (flag == 1);

	//CUDA_CALL(cudaMemcpy(col, label , sizeof(int)*MAX_HEIGHT * MAX_WIDTH, cudaMemcpyDeviceToHost));

	//CUDA_CALL(cudaMemcpy(img_local, img,w*h*sizeof(unsigned char), cudaMemcpyDeviceToHost));
	
	getLabelNum<<<1,h>>>(w,h,img,byF,label,d_label_num);
	CUDA_CALL(cudaMemcpy(numLabels, d_label_num, sizeof(int), cudaMemcpyDeviceToHost));

	//cudaMalloc(&label_index, *numLabels*sizeof(int));
	//cudaMemset(&label_index, -1, *numLabels*sizeof(int));

	//getNewLabel<<<1,h>>>(w,h,img,byF,label);

	resortLabel<<<1,1>>>(w,h,label,img,byF);

	setImg<<<1,h>>>(w,h,ws,wd,img,byF,label,imgOut);

	testOnCuda<<<1,1>>>(w,h,ws,wd,img,imgOut,byF,label,edge_num);

	cudaFree(d_flag);
	cudaFree(d_label_num);
	cudaFree(d_edge_num);
	cudaFree(label);

	return 0;
}
 