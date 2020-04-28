

#ifndef _BICUBICTEXTURE_KERNEL_CUH_
#define _BICUBICTEXTURE_KERNEL_CUH_


// render image using normal bilinear texture lookup
__global__ 
void d_render(uchar4 *d_output, uint width, uint height, float tx, float ty, float scale, float cx, float cy)
{
	uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	uint i = __umul24(y, width) + x;

	float u = (x - cx)*scale + cx + tx;
	float v = (y - cy)*scale + cy + ty;

	if ((x < width) && (y < height))
	{
		// write output color
		float c = tex2D(tex, u, v);
		d_output[i] = make_uchar4(c * 0xff, c * 0xff, c * 0xff, 0);
	}
}

#endif // _BICUBICTEXTURE_KERNEL_CUH_
