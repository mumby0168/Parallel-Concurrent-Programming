#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"

class sphere {
public:
	sphere() 
	{
		color = make_uchar4(0, 0, 128, 10);
		auto x = rand() % 100;
		auto y = rand() % 100;
		auto z = rand() % 100;

		center = vec3(x / 100.0, y / 100.0, z / 100.0);
		radius = 0.1;
	}


	__device__ void move(float x, float y, float z);
	__device__ void update_position(float x, float y, float z);
	__device__ bool hit(const ray& r, float tmin, float tmax) const;
	__device__ void set_brightness(float brightness);

	vec3 center;
	vec3 previous_center;
	uchar4 color;
	float radius;
};



__device__ bool sphere::hit(const ray& r, float t_min,
	float t_max) const
{	
	vec3 oc = r.origin() - center;
	float a = dot(r.direction(), r.direction());
	float b = dot(oc, r.direction());
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - a * c;
	if (discriminant > 0) {
		float temp = (-b - sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) {			
			return true;
		}
		temp = (-b + sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) {			
			return true;
		}
	}
	return false;
}

inline __device__ void sphere::set_brightness(float brightness)
{
	this->color.w = brightness;
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