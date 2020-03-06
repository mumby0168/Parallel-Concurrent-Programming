/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#ifndef _BICUBICTEXTURE_CU_
#define _BICUBICTEXTURE_CU_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <helper_math.h>

 // includes, cuda
#include <helper_cuda.h>

typedef unsigned int uint;
typedef unsigned char uchar;

cudaArray *d_imageArray = 0;

extern "C"
void initTexture(int imageWidth, int imageHeight, uchar *h_data)
{
	cudaResourceDesc texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = d_imageArray;
}

extern "C"
void freeTexture()
{
	checkCudaErrors(cudaFreeArray(d_imageArray));
}

__global__ void
d_render(uchar4 *d_output, uint width, uint height)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint i = y * width + x;


	if ((x < width) && (y < height))
	{
		/*Draws a cicle using pythag*/
		/*int centerX = width / 2;
		int centerY = height / 2;
		int radiusOfCirlce = 200;*/
		/*x -= centerX;
		y -= centerY;

		int length = sqrtf(x*x + y * y);

		int c = 0;

		if (length < radiusOfCirlce)
			c = 255;*/

		/*Draw Big Squares*/
		/*int c = ((((x & 0x20) == 0) ^ ((y & 0x20)) == 0)) * 255;*/

		float u = x / (float)width;
		float v = y / (float)height;
		u = 2.0*u - 1.0;
		v = -(2.0*v - 1.0);

		u *= width / height;

		float radius = 0.1;
		int c = 0;

		float distance = u * u + v * v;
		if (distance < radius * 2)
		{
			c = 255;
		}

		
		d_output[i] = make_uchar4(c, 0, 0, 0);		
		
	}
}


// render image using CUDA
extern "C"
void render(int width, int height,
	dim3 blockSize, dim3 gridSize, uchar4 *output)
{
	// call CUDA kernel, writing results to PBO memory
	d_render << <gridSize, blockSize >> > (output, width, height);

	getLastCudaError("kernel failed");
}

#endif
