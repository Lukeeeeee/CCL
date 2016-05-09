// cvLabelingImageLab: an impressively fast labeling routine for OpenCV
// Copyright (C) 2009 - Costantino Grana and Daniele Borghesani
//
// This library is free software; you can redistribute it and/or modify it under
// the terms of the GNU Lesser General Public License as published by the Free 
// Software Foundation; either version 3 of the License, or (at your option) 
// any later version.
//
// This library is distributed in the hope that it will be useful, but WITHOUT 
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more 
// details.
//
// You should have received a copy of the GNU Lesser General Public License along
// with this library; if not, see <http://www.gnu.org/licenses/>.
//
// For further information contact us at
// Costantino Grana/Daniele Borghesani - University of Modena and Reggio Emilia - 
// Via Vignolese 905/b - 41100 Modena, Italy - e-mail: {name.surname}@unimore.it

/**********************************************************************************************
***********************************************************************************************
#cat: icvLabelImage - CCL CPU alg for Demo                       

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
int icvLabelImage(int w, int h, int ws, int wd, unsigned char *img, char *imgOut, unsigned char byF,int *numLabels);





