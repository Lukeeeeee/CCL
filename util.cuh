/*************************************************************************************
LIBRARY: Connected-Component Labeling (CCL)
FILE:    util.cuh
DATE:    2/11/2014
UPDATED: 

Contains the utilities of word processing and image processing.
**************************************************************************************
ROUTINES:
stringRemoveDelimiter()
getCmdLineArgumentString()
read_bmp_24()
write_bmp_24()
*************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64)
#include <windows.h>
#else
#include <time.h>
#include <assert.h>
#include <sys/time.h>
#endif

/* Word processing */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64)
  #ifndef _CRT_SECURE_NO_DEPRECATE
  #define _CRT_SECURE_NO_DEPRECATE
  #endif
  #ifndef STRCASECMP
  #define STRCASECMP  _stricmp
  #endif
  #ifndef STRNCASECMP
  #define STRNCASECMP _strnicmp
  #endif
  #ifndef STRCPY
  #define STRCPY(sFilePath, nLength, sPath) strcpy_s(sFilePath, nLength, sPath)
  #endif

  #ifndef FOPEN
  #define FOPEN(fHandle,filename,mode) fopen_s(&fHandle, filename, mode)
  #endif
  #ifndef FOPEN_FAIL
  #define FOPEN_FAIL(result) (result != 0)
  #endif
  #ifndef SSCANF
  #define SSCANF sscanf_s
  #endif
#else
  #include <string.h>
  #include <strings.h>
  
  #ifndef STRCASECMP
  #define STRCASECMP  strcasecmp
  #endif
  #ifndef STRNCASECMP
  #define STRNCASECMP strncasecmp
  #endif
  #ifndef STRCPY
  #define STRCPY(sFilePath, nLength, sPath) strcpy(sFilePath, sPath)
  #endif

  #ifndef FOPEN
  #define FOPEN(fHandle,filename,mode) (fHandle = fopen(filename, mode))
  #endif
  #ifndef FOPEN_FAIL
  #define FOPEN_FAIL(result) (result == NULL)
  #endif
  #ifndef SSCANF
  #define SSCANF sscanf
  #endif
#endif

/**********************************************************************************************
***********************************************************************************************
#cat: getCmdLineArgumentString - Resolve the input arguments in command line with the form:
#cat:                            -"string_ref"="xxx"

Input:
argc, argv           - command line input arguments
string_ref           - retrieve the string after this word

Output:
string_retval        - retrieved string


Return Codes:
True     - found the word
False    - didn't find the word
**********************************************************************************************/
inline int stringRemoveDelimiter(char delimiter, const char *string)
{
    int string_start = 0;

    while (string[string_start] == delimiter)
    {
        string_start++;
    }

    if (string_start >= (int)strlen(string)-1)
    {
        return 0;
    }

    return string_start;
}

inline bool getCmdLineArgumentString(const int argc, const char **argv,
                                     const char *string_ref, char **string_retval)
{
    bool bFound = false;

    if (argc >= 1)
    {
        for (int i=1; i < argc; i++)
        {
            int string_start = stringRemoveDelimiter('-', argv[i]);
            char *string_argv = (char *)&argv[i][string_start];
            int length = (int)strlen(string_ref);

            if (!STRNCASECMP(string_argv, string_ref, length))
            {
                *string_retval = &string_argv[length+1];
                bFound = true;
                continue;
            }
        }
    }

    if (!bFound)
    {
        *string_retval = NULL;
    }

    return bFound;
}

/* Image processing */

#if !defined(WIN32) && !defined(_WIN32) && !defined(_WIN64)
typedef unsigned short WORD;
typedef unsigned int DWORD;
typedef unsigned int LONG;
#pragma pack(2)
typedef struct tagBITMAPFILEHEADER
{
    WORD bfType;
    DWORD bfSize;
    WORD bfReserved1;
    WORD bfReserved2;
    DWORD bfOffBits;
} BITMAPFILEHEADER;
#pragma pack()

typedef struct tagBITMAPINFOHEADER{
    DWORD biSize;
    LONG biWidth;
    LONG biHeight;
    WORD biPlanes;
    WORD biBitCount;
    DWORD biCompression;
    DWORD biSizeImage;
    LONG biXPelsPerMeter;
    LONG biYPelsPerMeter;
    DWORD biClrUsed;
    DWORD biClrImportant;
} BITMAPINFOHEADER;
#endif
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
int read_bmp_24(char *fname, char **ppdata, BITMAPFILEHEADER* pbitmapfileheader, BITMAPINFOHEADER* pbitmapinfoheader);

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
int write_bmp_24(char *fname, char *pdata, BITMAPFILEHEADER* pbitmapfileheader, BITMAPINFOHEADER* pbitmapinfoheader);

