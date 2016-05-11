/*************************************************************************************
LIBRARY: Connected-Component Labeling (CCL)
FILE:    main.cu
DATE:    2/11/2014
UPDATED: 

Contains the main function of ccl driver. The input of the system are several 
black-white bmp image with several targets for CCL, in which the background is
black and the foreground is white. A component is one pixel in white (foreground).
A target means all the components connected (if two components are separated by
background, they belong to different target).

A CPU version CCL module has been called in the main function for demo and 
comperation. What you need to do is designing a GPU version to achieve as much
speed as you can (no need to follow the same alg of CPU). The interface of the 
gpu module has been defined in ccl_gpu.cuh. 

The output of the system is a bmp image with 24bit true color image for verification.   
*************************************************************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "util.cuh"
#include "ccl_cpu.cuh" 
#include "ccl_gpu.cuh"
#include <stdio.h>
#include <math.h>



int main(int argc, const char* argv[])
{

    /* Word processing */
    cudaError_t cudaStatus; 
	char* inputfile = "3_21.bmp"; //input file name
	char* cpu_outputfile = "cpu.bmp"; //output file name of CPU alg
	char* gpu_outputfile = "gpu.bmp"; //output file name of GPU alg
	getCmdLineArgumentString(argc, argv, "infile", &inputfile); 
	getCmdLineArgumentString(argc, argv, "cpu_outfile", &cpu_outputfile);
	getCmdLineArgumentString(argc, argv, "gpu_outfile", &gpu_outputfile);
	printf("inputfile=%s,cpu_outputfile=%s,gpu_outputfile=%s\n",inputfile,cpu_outputfile,gpu_outputfile);

	/* Image input */
	BITMAPFILEHEADER fileHeader; //header of bmp image
    BITMAPINFOHEADER infoHeader;
	char *ori_data; //bmp image data
	//only accept input image with depth equal to 24bit no padding
	read_bmp_24(inputfile,&ori_data,&fileHeader,&infoHeader);
	printf("%d %d\n",infoHeader.biWidth, infoHeader.biHeight); //ÐÞ¸Ä
	unsigned char *pidata, *d_pidata; //input data for CPU and GPU alg
	pidata = (unsigned char*)malloc(infoHeader.biWidth*infoHeader.biHeight*sizeof(unsigned char));
	for(int i = 0; i < infoHeader.biWidth*infoHeader.biHeight; i++)
	{
		pidata[i] = (ori_data[i*3] ? 1 : 0); //set the foreground to 1, background to 0
	}
	//copy to GPU 
	cudaMalloc(&d_pidata,infoHeader.biWidth*infoHeader.biHeight*sizeof(unsigned char));
	//cudaMalloc(&d_pidata,1024*1024*1024);
	cudaMemcpy(d_pidata,pidata,infoHeader.biWidth*infoHeader.biHeight*sizeof(unsigned char),cudaMemcpyHostToDevice);

	/* Output Initialization */
	int *podata, *d_podata;
	podata = (int*)malloc(infoHeader.biWidth*infoHeader.biHeight*sizeof(int));
	cudaMalloc(&d_podata,infoHeader.biWidth*infoHeader.biHeight*sizeof(int));
	printf("malloc memory %d \n",infoHeader.biWidth*infoHeader.biHeight*sizeof(int));
	memset(podata,0,infoHeader.biWidth*infoHeader.biHeight*sizeof(int));
	cudaMemset(d_podata,0,infoHeader.biWidth*infoHeader.biHeight*sizeof(int));
	int iNumLabels = -1;

	/* CPU CCL the BBDT proposed by C. Grana, D. Borghesani, R. Cucchiara*/
#if defined(WIN32) || defined(_WIN32) || defined(WIN64)
	LARGE_INTEGER m_nFreq;  
    LARGE_INTEGER m_nBeginTime;  
    LARGE_INTEGER nEndTime;  
    QueryPerformanceFrequency(&m_nFreq); 
    QueryPerformanceCounter(&m_nBeginTime); 

	icvLabelImage(infoHeader.biWidth,infoHeader.biHeight, infoHeader.biWidth, infoHeader.biWidth*4, pidata,(char*)podata, 1, &iNumLabels);

	QueryPerformanceCounter(&nEndTime);  
	printf("CPU Elapse time: %fms\n",(double)(nEndTime.QuadPart-m_nBeginTime.QuadPart)*1000/m_nFreq.QuadPart);
	printf("Num of Labels = %d\n",iNumLabels);
#else
	struct timeval start,end;
	gettimeofday(&start,NULL);
	long long start_time = start.tv_sec*1000000+start.tv_usec;

	icvLabelImage(infoHeader.biWidth,infoHeader.biHeight, infoHeader.biWidth, infoHeader.biWidth*4, pidata,(char*)podata, 1, &iNumLabels);

	gettimeofday(&end,NULL);
	float end_time = (end.tv_sec*1000000+end.tv_usec - start_time)/1000.0;
	printf("CPU Elapse time: %fms\n",end_time);
	printf("Num of Labels = %d\n",iNumLabels);
#endif

	/* GPU CCL of yours */
	iNumLabels = -1;
#if defined(WIN32) || defined(_WIN32) || defined(WIN64)
	unsigned int *gpu_block;
	QueryPerformanceFrequency(&m_nFreq); 
    QueryPerformanceCounter(&m_nBeginTime);

	cudaMalloc((void **)&gpu_block,1000);

	gpuLabelImage(infoHeader.biWidth,infoHeader.biHeight, infoHeader.biWidth, infoHeader.biWidth*4, d_pidata, d_podata, 1, &iNumLabels);
	
	cudaDeviceSynchronize();


	QueryPerformanceCounter(&nEndTime);  
	printf("GPU Elapse time: %fms\n",(double)(nEndTime.QuadPart-m_nBeginTime.QuadPart)*1000/m_nFreq.QuadPart);
	printf("Num of Labels = %d\n",iNumLabels);
#else
	gettimeofday(&start,NULL);
	start_time = start.tv_sec*1000000+start.tv_usec;

	gpuLabelImage(infoHeader.biWidth,infoHeader.biHeight, infoHeader.biWidth, infoHeader.biWidth*4, d_pidata, d_podata, 1, &iNumLabels);

	cudaDeviceSynchronize();
	gettimeofday(&end,NULL);
	end_time = (end.tv_sec*1000000+end.tv_usec - start_time)/1000.0;
	printf("GPU Elapse time: %fms\n",end_time);
	printf("Num of Labels = %d\n",iNumLabels);
#endif
	
	/* Image output and result checking */
	for(int i = 0; i < infoHeader.biWidth*infoHeader.biHeight; i++)
	{
		ori_data[i*3] = (podata[i]*23)%256;
		ori_data[i*3 + 1] = (podata[i]*30)%256;
	}
	write_bmp_24(cpu_outputfile,ori_data,&fileHeader,&infoHeader);

	cudaMemcpy(podata,d_podata,infoHeader.biWidth*infoHeader.biHeight*sizeof(int),cudaMemcpyDeviceToHost);
	for(int i = 0; i < infoHeader.biWidth*infoHeader.biHeight; i++)
	{
		ori_data[i*3] = (podata[i]*23)%256;
		ori_data[i*3 + 1] = (podata[i]*30)%256;
	}
	write_bmp_24(gpu_outputfile,ori_data,&fileHeader,&infoHeader);
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
	printf("%s\n",cudaGetErrorString(cudaGetLastError()));
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
	
	cudaFree(d_pidata);
	cudaFree(d_podata);

	system("pause");
    return 0;
}

