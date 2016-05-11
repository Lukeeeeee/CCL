#define CUDA_CALL(x) {const cudaError_t a = (x);if (a != cudaSuccess) { printf("\nCuda error: %s (err_num = %d)\n",cudaGetErrorString(a),a);cudaDeviceReset();}}
#define MAX_HEIGHT 1024
#define MAX_WIDTH 1024
//#define MAX_VERTEX MAX_WIDTH * MAX_HEIGHT

#define INT_PTR(x) (*((int*)(&(x))))

__device__ int dx[8] = {-1,0,1,-1,1,-1,0,1};
__device__ int dy[8] = {-1,-1,-1,0,0,1,1,1};

__device__ int get_loc();
__device__ int get_x();
__device__ int get_y();
__device__ bool check_bound(int x, int y, int w, int h);
__device__ bool check_connect(int loc1, int loc2, unsigned char *img, unsigned char byF)