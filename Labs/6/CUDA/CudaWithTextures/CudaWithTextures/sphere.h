#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"

class sphere {
public:
	sphere() 
	{
		color = make_uchar4(0, 0, 128, 10);
		center = vec3(0.5, 0.5, 0.5);
		radius = 0.2;
	}
	__device__ sphere(vec3 cen, float r) : center(cen), radius(r) {};
	__device__ bool hit(const ray& r, float tmin, float tmax) const;
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


#endif