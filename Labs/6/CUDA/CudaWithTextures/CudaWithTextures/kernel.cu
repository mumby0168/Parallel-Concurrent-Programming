


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


#define PARTICLE_COUNT 1

 // includes, cuda
#include <helper_cuda.h>

typedef unsigned int uint;
typedef unsigned char uchar;

sphere spheres[PARTICLE_COUNT];

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
//__device__ vec3 castRay(const ray& r, const sphere *spheres) {
//	hit_record rec;
//	for (int i = 0; i < PARTICLE_COUNT; i++)
//	{
//		if (spheres[i].hit(r, 0.0, FLT_MAX)) {
//
//			//TODO: Pick a colour to render when hit
//			return vec3(0.5, 0.5, 0.5);
//		}
//		else {
//			//TODO: render the background colour
//			return vec3(1, 1, 1);
//		}
//	}
//	
//}

__device__ static int ticks = 1;

__device__ static float xStep = 0.01;
__device__ static float xPos = 0;

__device__ static float yStep = 0.01;
__device__ static float yPos = 0;

__device__ static float zStep = 0.001;
__device__ static float zPos = 0;


extern "C" void init_particles()
{
	for (int i = 0; i < PARTICLE_COUNT; i++)
	{
		spheres[i] = sphere();
	}
}



__global__ void move_particles()
{

}

__global__ void bound_particles()
{

}

__global__ void colour_particles()
{

}


////TODO: Possibly change this to check every particles boundaries.
//__global__ void create_world(hitable **d_list, hitable **d_world) {
//	
//	if (threadIdx.x == 0 && blockIdx.x == 0) {
//		*(d_list) = new sphere(vec3(xPos,yPos,zPos), 0.2);	
//		*d_world = new hitable_list(d_list, 1);
//	}
//}

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
d_render(uchar4 *d_output, uint width, uint height, const sphere *spheres)
{
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
	vec3 eye = vec3(0.5, 0.5, 1.5);
	float distFrEye2Img = 1.0;
	if ((x < width) && (y < height))
	{		
		//for each pixel
		vec3 pixelPos = vec3(u, v, eye.z() - distFrEye2Img);
		//fire a ray:
		ray r;
		r.Origin = eye;
		r.Direction = pixelPos - eye; //view direction along negtive z-axis!
		for (int j = 0; j < PARTICLE_COUNT; j++)
		{			

			

			if (spheres[j].hit(r, 0.0, FLT_MAX))
			{				
				//TODO: This may not be best solution as a particle behind could be rendered first. i.e don't return.
				d_output[i] = spheres[j].color;		
				return;
			}
			else
			{
				d_output[i] = make_uchar4(124, 252, 0, 0);
			}
		}
	}			
}


// render image using CUDA
extern "C" 
void render(int width, int height, dim3 blockSize, dim3 gridSize, uchar4 *output)
{
	sphere *d_spheres = 0;

	checkCudaErrors(cudaMalloc((void **)&d_spheres, PARTICLE_COUNT * sizeof(sphere)));

	checkCudaErrors(cudaMemcpy(d_spheres, spheres, PARTICLE_COUNT * sizeof(sphere), cudaMemcpyHostToDevice));

	
	// call CUDA kernel, writing results to PBO memory
	d_render << <gridSize, blockSize >> > (output, width, height, d_spheres);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(spheres, d_spheres, PARTICLE_COUNT * sizeof(sphere), cudaMemcpyDeviceToHost));


	getLastCudaError("kernel failed");
}

#endif
