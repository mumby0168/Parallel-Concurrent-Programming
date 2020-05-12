#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"

class sphere {
public:
	sphere() 
	{
		color = make_uchar4(255, 0, 0, 0);
		auto x = rand() % 100;
		auto y = rand() % 100;
		auto z = rand() % 100;

		center = vec3(x, y, z);
		radius = 0.01;
	}


	__device__ void move(float x, float y, float z);
	__device__ void update_position(float x, float y, float z);
	__device__ bool hit(const vec3 direction) const;
	__device__ void set_brightness(float brightness);
	__device__ void solid_colour();
	__device__ void norm();

	vec3 center;
	vec3 normCenter;
	vec3 previous_center;
	uchar4 color;
	float radius;
};



__device__ bool sphere::hit(const vec3 direction) const
{	
	vec3 oc = this->normCenter;
	float a = dot(direction, direction);
	float b = dot(oc, direction);
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - a * c;
	return (discriminant > 0);
}

inline __device__ void sphere::set_brightness(float brightness) {
	unsigned char bright = (unsigned char)brightness;
	this->color = make_uchar4(brightness, this->color.y, this->color.z, 255);
}

inline __device__ void sphere::solid_colour()
{
	this->color = make_uchar4(255, 0, 0, 255);
}

inline __device__ void sphere::norm()
{
	float normx = 2.0 * (center[0] / 100.0f) - 1.0;
	float normy = 2.0 * (center[1] / 100.0f) - 1.0;
	float normz = 2.0 * (center[2] / 100.0f) - 3.0;
	normCenter = vec3(normx, normy, normz);
}

__device__ void sphere::move(float x, float y, float z) {	
	previous_center = center;
	center = vec3(x + center.x(), y + center.y(), z + center.z());
}

__device__ void sphere::update_position(float x, float y, float z) {
	previous_center = center;
	center = vec3(x, y, z);
}



#endif