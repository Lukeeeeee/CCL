/*************************************************************************************
LIBRARY: Connected-Component Labeling (CCL)
FILE:    main.cu
DATE:    2/11/2014
UPDATED: 

Contains the interface of CCL CPU alg.
**************************************************************************************
ROUTINES:
gpuLabelImage()
*************************************************************************************/



/**********************************************************************************************
***********************************************************************************************
#cat: gpuLabelImage - CCL CPU alg interface                       

Input:
w           - width of the image in pixels
h           - height of the image in pixels
ws          - pitch of the source image in bytes
wd          - pitch of the destination image in bytes 
img         - source image
byF         - foreground mark (always 1 in this driver)

Output:
numLabels   - The number of Labels (targets) in the image
imgOut      - destination image

Return Codes:
reserved
**********************************************************************************************/

int gpuLabelImage(int w, int h, int ws, int wd, unsigned char *img, int *imgOut, unsigned char byF,int *numLabels);