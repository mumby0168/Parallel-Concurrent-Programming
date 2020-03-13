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
#include "Ray.h"
#include "hitable.h"
#include "hitable_list.h"
#include "sphere.h"
#include "vec3.h"

#include <helper_math.h>

 // includes, cuda
#include <helper_cuda.h>

typedef unsigned int uint;
typedef unsigned char uchar;

cudaArray *d_imageArray = 0;

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const
	file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << "at " << file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}
__device__ vec3 castRay(const ray& r, hitable **world) {
	hit_record rec;
	if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {

		//TODO: Pick a colour to render when hit
		return vec3(0.5, 0.5, 0.5);
	}
	else {
		//TODO: render the background colour
		return vec3(1, 1, 1);
	}
}

__device__ static int ticks = 1;

__device__ static float xStep = 0.01;
__device__ static float xPos = 0;

__device__ static float yStep = 0.01;
__device__ static float yPos = 0;

__device__ static float zStep = 0.001;
__device__ static float zPos = 0;

__global__ void create_world(hitable **d_list, hitable **d_world) {
	
	xPos += xStep;
	zPos += zStep;
	yPos += yStep;

	if (xPos > 1)
	{		
		xStep = (-xStep);		
	}
	if (xPos < -1)
	{
		xStep = 0.01;
	}
	if (zPos > 1)
	{
		zStep = (-zStep);
	}
	else if (zPos < -1)
	{
		zStep = 0.01;
	}
	if (yPos > 1)
	{
		yStep = (-yStep);
	}
	else if (yPos < -1)
	{
		yStep = 0.01;
	}


	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*(d_list) = new sphere(vec3(xPos,yPos,zPos), 0.2);	
		*d_world = new hitable_list(d_list, 1);
	}
}
__global__ void free_world(hitable **d_list, hitable **d_world) {
	delete *(d_list);
	delete *(d_list + 1);
	delete *d_world;
}

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
d_render(uchar4 *d_output, uint width, uint height, hitable **d_world)
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
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint i = y * width + x;
	

	float u = x / (float)width;
	float v = y / (float)height;
	u = 2.0*u - 1.0;
	v = -(2.0*v - 1.0);

	u *= width / height;

	u *= 2.0;
	v *= 2.0;
	vec3 eye = vec3(0, 0.5, 1.5);
	float distFrEye2Img = 1.0;;
	if ((x < width) && (y < height))
	{
		//for each pixel
		vec3 pixelPos = vec3(u, v, eye.z() - distFrEye2Img);
		//fire a ray:
		ray r;
		r.Origin = eye;
		r.Direction = pixelPos - eye; //view direction along negtive z-axis!
		vec3 col = castRay(r, d_world);
		float red = col.x();
		float green = col.y();
		float blue = col.z();
		d_output[i] = make_uchar4(red * 255, green * 255, blue * 255, 0);
	}
		

		/*float radius = 0.1;
		int c = 0;

		float distance = u * u + v * v;
		if (distance < radius * 2)
		{
			c = 255;
		}

		
		d_output[i] = make_uchar4(c, 0, 0, 0);		*/
			
}


// render image using CUDA
extern "C"
void render(int width, int height,
	dim3 blockSize, dim3 gridSize, uchar4 *output)
{
	hitable **d_list;
	checkCudaErrors(cudaMalloc((void **)&d_list, 2 * sizeof(hitable *)));
	hitable **d_world;
	checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
	create_world << <1, 1 >> > (d_list, d_world);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	// call CUDA kernel, writing results to PBO memory
	d_render << <gridSize, blockSize >> > (output, width, height, d_world);

	getLastCudaError("kernel failed");
}

#endif
