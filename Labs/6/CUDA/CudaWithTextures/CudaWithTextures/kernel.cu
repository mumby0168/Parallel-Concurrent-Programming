


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


#define PARTICLE_COUNT 50

 // includes, cuda
#include <helper_cuda.h>

typedef unsigned int uint;
typedef unsigned char uchar;
using namespace std::chrono;

sphere spheres[PARTICLE_COUNT];
vec3 randoms[PARTICLE_COUNT];

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
	return r / 100;		
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



__global__ void move_particles(sphere *spheres, const vec3 *randoms, const int *delta)
{	
	int i = threadIdx.x;
	float d = (*delta) / 1000.0;
	spheres[i].move(randoms[i].x() * d, randoms[i].y() * d, randoms[i].z() * d);
}

__global__ void apply_gravity(sphere *spheres, const int *delta)
{
	int i = threadIdx.x;
	float d = (*delta) / 1000.0;
	spheres[i].move(0,-0.9 * d,0);
}

__global__ void bound_particles(sphere *spheres)
{
	int i = threadIdx.x;
	int x = spheres[i].center.x();
	int y = spheres[i].center.y();
	int z = spheres[i].center.z();	
	bool update = false;

	if (x > 1) {
		x = -1;
		update = true;
	}
	if (x < -1) {
		x = 1;
		update = true;
	}

	if (y > 1) {
		y = -1;
		update = true;
	}
	if (y< -1) {
		y= 1;
		update = true;
	}
	if (z > 1) {
		z = -1;
		update = true;
	}
	if (z < -1) {
		z = 1;
		update = true;
	}

	if(update)
		spheres[i].update_position(x, y, z);

}


__global__ void colour_particles(const ColorMode *mode, sphere *spheres)
{
	int i = threadIdx.x;

	if (*mode == Solid) {
		spheres[i].solid_colour();
	}
	else if (*mode == CenterMass) {
		
	}
	else if (*mode == Speed) {

		float toRoute = pow((spheres[i].center.x() - spheres[i].previous_center.x()), 2) +
			pow((spheres[i].center.y() - spheres[i].previous_center.y()), 2) + pow((spheres[i].center.z() - spheres[i].previous_center.z()), 2);

		float distance = sqrt(toRoute);

		float percentage = (1.0 / distance) / 100;

		spheres[i].set_brightness(255 * percentage);
	}
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
	
	if ((x < width) && (y < height))
	{		
		//for each pixel
		
		//fire a ray:
		ray r = ray(u, v);		
		for (int j = 0; j < PARTICLE_COUNT; j++)
		{				
			if (spheres[j].hit(r, 0.0, FLT_MAX))
			{				
				//TODO: This may not be best solution as a particle behind could be rendered first. i.e don't return.
				d_output[i] = spheres[j].color;		
				return;
			}			
		}
		d_output[i] = make_uchar4(220, 220, 220, 255);
	}			
}


// render image using CUDA
extern "C" 
void render(int width, int height, dim3 blockSize, dim3 gridSize, uchar4 *output, int deltaTime, ColorMode mode)
{
	sphere *d_spheres = 0;
	vec3 *d_randoms = 0;
	int *d_DeltaTime = 0;
	ColorMode *d_mode = 0;

	update_randoms();

	checkCudaErrors(cudaMalloc((void **)&d_randoms, PARTICLE_COUNT * sizeof(vec3)));

	checkCudaErrors(cudaMemcpy(d_randoms, randoms, PARTICLE_COUNT * sizeof(vec3), cudaMemcpyHostToDevice));


	checkCudaErrors(cudaMalloc((void **)&d_spheres, PARTICLE_COUNT * sizeof(sphere)));

	checkCudaErrors(cudaMemcpy(d_spheres, spheres, PARTICLE_COUNT * sizeof(sphere), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **)&d_DeltaTime, sizeof(int)));

	checkCudaErrors(cudaMemcpy(d_DeltaTime, &deltaTime, sizeof(int), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **)&d_mode, sizeof(ColorMode)));

	checkCudaErrors(cudaMemcpy(d_mode, &mode, sizeof(ColorMode), cudaMemcpyHostToDevice));

	
	move_particles<<<1, 50>>>(d_spheres, d_randoms, d_DeltaTime);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	if (gravity_enabled)
	{
		apply_gravity<<<1, 50 >>>(d_spheres, d_DeltaTime);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}

	bound_particles<<<1, 50>>>(d_spheres);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	colour_particles<<<1,50>>>(d_mode, d_spheres);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	d_render<<<gridSize, blockSize>>>(output, width, height, d_spheres);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(spheres, d_spheres, PARTICLE_COUNT * sizeof(sphere), cudaMemcpyDeviceToHost));


	getLastCudaError("kernel failed");
}

#endif
