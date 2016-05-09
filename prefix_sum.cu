#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <ctime>
#include <cstdlib>

__device__ int get_loc1() {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	return gridDim.x * blockDim.x * idy +idx;
}

__device__ int get_x1() {
	return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ int get_y1() {
	return blockIdx.y * blockDim.y + threadIdx.y;
}

int next_pow_2(int n) {
	n--;
	n = n>>1 | n;
	n = n>>2 | n;
	n = n>>4 | n;
	n = n>>8 | n;
	n = n>>16 | n;
	return ++n;
}

__global__ void get_new_data(int count, int *d_data, unsigned char *img, unsigned char byF) {
	int loc = get_loc1();
	if(loc>=count) return;
	d_data[loc] = (d_data[loc] == loc && img[loc] == byF) ? 1:0;
}

__global__ static void compute_prefix_sum(int count, int *d_data, int cnt2) {
	extern __shared__ unsigned int shared_mem[];
	int loc = get_loc1();
	shared_mem[loc] = (loc < count) ? d_data[loc] : 0;
	__syncthreads();

	for(unsigned int s=cnt2/2;s>0;s>>=1) {
		if (loc < s) {
			shared_mem[loc] += shared_mem[loc+s];
		}
		__syncthreads();
	}
	if(loc == 0) {
		d_data[0] = shared_mem[0];
	}


}
int get_prefix_sum(int w, int h, int *label, unsigned char *img, unsigned char byF) {

	int count = w*h;

	int *d_data;

	cudaMalloc(&d_data, sizeof(int)*count);
	cudaMemcpy(d_data, label, sizeof(int)*count, cudaMemcpyHostToDevice);

	int npt_count = next_pow_2(count);
	int data_size = npt_count * sizeof(int);
	
	const dim3 get_new_data_blocks_rect(w/32+1,h/32+1);
	const dim3 get_new_data_threads_rect(32,32);

	const dim3 compute_prefix_sum_blocks_rect(w/32+1, h/32+1);
	const dim3 compute_prefix_sum_threads_rect(32, 32);

	get_new_data<<<get_new_data_blocks_rect, get_new_data_threads_rect>>>(count, d_data, img, byF);
	compute_prefix_sum<<<compute_prefix_sum_threads_rect, compute_prefix_sum_blocks_rect, data_size>>>(count, d_data, npt_count);
	
	int sum;

	cudaMemcpy(&sum, d_data, sizeof(int)*count, cudaMemcpyDeviceToHost);
	cudaFree(d_data);
	 
	return sum;

}