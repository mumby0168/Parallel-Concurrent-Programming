#ifndef RAYH
#define RAYH
#include "vec3.h"

class ray
{
	private:
		const vec3 eye = vec3(0.5, 0.5, 5);
		const float distFrEye2Img = 1.0;



    public:
        __device__ ray(float u, float v) {
			vec3 pixelPos = vec3(u, v, eye.z() - distFrEye2Img);
			this->Origin = eye;
			this->Direction = pixelPos - eye;
		}
        __device__ ray(const vec3& a, const vec3& b) { Origin = a; Direction = b; }
        __device__ vec3 origin() const       { return Origin; }
        __device__ vec3 direction() const    { return Direction; }
        __device__ vec3 point_at_parameter(float t) const { return Origin + t*Direction; }		

        vec3 Origin;
        vec3 Direction;
};

#endif