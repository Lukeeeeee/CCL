/*************************************************************************************
LIBRARY: Connected-Component Labeling (CCL)
FILE:    main.cu
DATE:    2/11/2014
UPDATED: 

Contains the utilities of word processing and image processing.
*************************************************************************************/

#include "util.cuh"

/**********************************************************************************************
***********************************************************************************************
#cat: read_bmp_24 - Takes a grayscale input bmp image and distruct to header and data

Input:
fname               - name of the input bmp file

Output:
ppdata              - image data fetched from the image
pbitmapfileheader   - file header of bmp image
pbitmapinfoheader   - info header of bmp image

Return Codes:
Zero     - successful completion
Negative - system error
**********************************************************************************************/
int read_bmp_24(char *fname, char **ppdata, BITMAPFILEHEADER* pbitmapfileheader, BITMAPINFOHEADER* pbitmapinfoheader)
{
  unsigned char palette[8];
  FILE *f = fopen(fname,"rb");
  if(f == NULL) {
        printf("Can not find file %s\n",fname);
        return -1;
    }
  fread(pbitmapfileheader,sizeof(BITMAPFILEHEADER),1,f);
  fread(pbitmapinfoheader,sizeof(BITMAPINFOHEADER),1,f);
  char *pdata = (char*)malloc(pbitmapinfoheader->biSizeImage);
  fread(pdata,sizeof(char),pbitmapinfoheader->biSizeImage,f);
  *ppdata = pdata;

  return 0;
}

/**********************************************************************************************
***********************************************************************************************
#cat: write_bmp_24 - output a grayscale bmp image according to the input header info and data

Input:
fname               - name of the output bmp file

Output:
ppdata              - image data fetched from the image
pbitmapfileheader   - file header of bmp image
pbitmapinfoheader   - info header of bmp image

Return Codes:
Zero     - successful completion
Negative - system error
**********************************************************************************************/
int write_bmp_24(char *fname, char *pdata, BITMAPFILEHEADER* pbitmapfileheader, BITMAPINFOHEADER* pbitmapinfoheader)
{
	FILE *f = fopen(fname,"wb");
	if(f == NULL) {
        printf("Can not find file %s\n",fname);
        return -1;
    }
	fwrite(pbitmapfileheader,sizeof(BITMAPFILEHEADER),1,f);
	fwrite(pbitmapinfoheader,sizeof(BITMAPINFOHEADER),1,f);
	fwrite(pdata, sizeof(char),pbitmapinfoheader->biSizeImage,f);
	return 0;
}