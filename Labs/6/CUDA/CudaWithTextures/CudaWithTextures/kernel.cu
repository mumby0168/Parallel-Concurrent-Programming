


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
#include <time.h>
#include <chrono>
#include <helper_math.h>
#include "types.h"


#define PARTICLE_COUNT 150

 // includes, cuda
#include <helper_cuda.h>

typedef unsigned int uint;
typedef unsigned char uchar;
using namespace std::chrono;

sphere spheres[PARTICLE_COUNT];
vec3 randoms[PARTICLE_COUNT];
uint threadPerBlock = 25;
uint blocks = PARTICLE_COUNT / threadPerBlock;

bool gravity_enabled = false;

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

extern "C" void set_gravity(bool value) {
	printf("gravity %d\n", value);
	gravity_enabled = value;
}


float generate_random() {

	float r = rand() % 100;	
	if(rand() % 100 < 50)
		r = -r;	
	return r / 5;
}


extern "C" void init_particles()
{
	milliseconds ms = duration_cast<milliseconds>(
		system_clock::now().time_since_epoch());

	srand(ms.count());
}


void update_randoms() {
	for (int i = 0; i < PARTICLE_COUNT; i++) {
		randoms[i] = vec3(generate_random(), generate_random(), generate_random());
	}
}



__global__ void move_particles(sphere *spheres, const vec3 *randoms, const SimulationParams *params)
{	
	int i = threadIdx.x + (blockDim.x * blockIdx.x);
	float d = (params->dt) / 1000.0;
	spheres[i].norm();
	spheres[i].move(randoms[i].x() * d, randoms[i].y() * d, randoms[i].z() * d);
}

__global__ void apply_gravity(sphere *spheres, const SimulationParams *params)
{
	int i = threadIdx.x + (blockDim.x * blockIdx.x);
	float d = (params->dt) / 1000.0;
	spheres[i].move(0,-9.0 * d,0);
}

__global__ void bound_particles(sphere *spheres)
{
	int i = threadIdx.x + (blockDim.x * blockIdx.x);
	int x = spheres[i].center.x();
	int y = spheres[i].center.y();
	int z = spheres[i].center.z();	
	bool update = false;

	if (x > 100) {
		x = 0;
		update = true;
	}
	if (x < 0) {
		x = 100;
		update = true;
	}

	if (y > 100) {
		y = 0;
		update = true;
	}
	if (y< 0) {
		y= 100;
		update = true;
	}
	if (z > 100) {
		z = 0;
		update = true;
	}
	if (z < 0) {
		z = 100;
		update = true;
	}

	if(update)
		spheres[i].update_position(x, y, z);

}


__global__ void colour_particles(SimulationParams *params, sphere *spheres)
{
	int i = threadIdx.x + (blockDim.x * blockIdx.x);

	float toRoute = pow((spheres[i].center.x() - spheres[i].previous_center.x()), 2) +
		pow((spheres[i].center.y() - spheres[i].previous_center.y()), 2) + pow((spheres[i].center.z() - spheres[i].previous_center.z()), 2);

	float distance = sqrt(toRoute);

	if (distance > params->max) {
		params->max = distance;
	}

	if (params->colorMode == Solid) {
		spheres[i].solid_colour();
	}
	else if (params->colorMode == CenterMass) {
		float toCenter = pow((spheres[i].center.x() - 50), 2) +
			pow((spheres[i].center.y() - 50), 2) + pow((spheres[i].center.z() - 50), 2);

		float distanceToCenter = sqrt(toCenter);

		float percentage = 1 - (distanceToCenter / params->maxCenterDistance);

		spheres[i].set_brightness(percentage * 255);
	}
	else if (params->colorMode == Speed) {

		float percentage = (distance / params->max);

		float color = 255 * percentage;

		spheres[i].set_brightness(color);
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


__device__ uchar4 get_colour(sphere *spheres, vec3 direction) {
	
}


__global__ void
d_render(uchar4 *d_output, uint width, uint height, sphere *spheres)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint i = y * width + x;		

	
	if ((x < width) && (y < height))
	{		
		//for each pixel

		float u = x / (float)width;
		float v = y / (float)height;
		u = 2.0*u - 1.0;
		v = -(2.0*v - 1.0);
		u *= width / height;
		u *= 2.0;
		v *= 2.0;
		
		const vec3 direction = vec3(u, v, -3);
		float a = dot(direction, direction);
		float r2 = 0.01 * 0.01;
		for (int j = 0; j < PARTICLE_COUNT; j++)
		{
			vec3 oc = spheres[j].normCenter;
			
			float b = dot(oc, direction);
			float c = dot(oc, oc) - r2;
			float discriminant = b * b - a * c;
			if (discriminant > 0) {
				d_output[i] = spheres[j].color;
				return;
			}
		}

		d_output[i] = make_uchar4(220, 220, 220, 255);
		
	}			
}

__global__ void free(sphere *d_spheres, vec3 * d_randoms, SimulationParams *d_params) {
	delete d_spheres;
	delete d_randoms;
	delete d_params;
}


// render image using CUDA
extern "C" 
void render(int width, int height, dim3 blockSize, dim3 gridSize, uchar4 *output, const SimulationParams params)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	sphere *d_spheres = 0;
	vec3 *d_randoms = 0;
	SimulationParams *d_params;

	

	update_randoms();

	checkCudaErrors(cudaMalloc((void **)&d_randoms, PARTICLE_COUNT * sizeof(vec3)));

	checkCudaErrors(cudaMemcpy(d_randoms, randoms, PARTICLE_COUNT * sizeof(vec3), cudaMemcpyHostToDevice));


	checkCudaErrors(cudaMalloc((void **)&d_spheres, PARTICLE_COUNT * sizeof(sphere)));

	checkCudaErrors(cudaMemcpy(d_spheres, spheres, PARTICLE_COUNT * sizeof(sphere), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **)&d_params, sizeof(SimulationParams)));

	checkCudaErrors(cudaMemcpy(d_params, &params, sizeof(SimulationParams), cudaMemcpyHostToDevice));

	cudaEventRecord(start, 0);

	move_particles<<<threadPerBlock, blocks>>>(d_spheres, d_randoms, d_params);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	if (gravity_enabled)
	{
		apply_gravity<<<threadPerBlock, blocks >>>(d_spheres, d_params);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}

	bound_particles<<<threadPerBlock, blocks>>>(d_spheres);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	colour_particles<<<threadPerBlock, blocks >>>(d_params, d_spheres);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("Taken: %f", elapsedTime);

	d_render<<<gridSize, blockSize>>>(output, width, height, d_spheres);

	

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(spheres, d_spheres, PARTICLE_COUNT * sizeof(sphere), cudaMemcpyDeviceToHost));

	getLastCudaError("kernel failed");


}

#endif
