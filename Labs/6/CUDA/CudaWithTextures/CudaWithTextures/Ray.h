#ifndef RAYH
#define RAYH
#include "vec3.h"

class ray
{
    public:
        __device__ ray() {}
        __device__ ray(const vec3& a, const vec3& b) { Origin = a; Direction = b; }
        __device__ vec3 origin() const       { return Origin; }
        __device__ vec3 direction() const    { return Direction; }
        __device__ vec3 point_at_parameter(float t) const { return Origin + t*Direction; }

        vec3 Origin;
        vec3 Direction;
};

#endif