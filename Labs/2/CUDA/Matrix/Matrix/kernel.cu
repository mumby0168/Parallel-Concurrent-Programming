
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size, unsigned int *blocks, unsigned int *threadsPerBlock);

void printArray(const int a[], int size);
void fillArray(int a[], int size);
void cleanUpMatrixOperation(int *pA, int *pB, int *pResult);
void addMatricesWithCuda(const int a[3][3], const int b[3][3], int c[3][3]);

__global__ 
void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x + (blockIdx.x * blockDim.x);

	c[i] = a[i] + b[i];
}

__global__
void addMarticesKernel(int **resultMatrix, const int **matrixA, const int **matrixB)
{
	int x = threadIdx.x + (threadIdx.y * blockDim.x);
	int y = threadIdx.y;

	resultMatrix[x][y] = matrixA[x][y] + matrixB[x][y];
}

void cudaAddingExample()
{
	const int arraySize = 50;
	int a[arraySize];
	int b[arraySize];
	unsigned int blocks = 1;
	unsigned int threadsPerBlock = 10;

	fillArray(a, arraySize);
	fillArray(b, arraySize);

	int c[arraySize] = { 0 };

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize, &blocks, &threadsPerBlock);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");		
	}


	printArray(a, arraySize);
	printf("\n+\n ");
	printArray(b, arraySize);
	printf(" \n= \n");
	printArray(c, arraySize);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");		
	}

}

void cudaMatrixExample()
{

	int width = 3;
	int height = 3;

	const int a[3][3] = {
		{5, 5, 5},
		{10, 10, 10},
		{6, 6, 6}
	};

	const int b[3][3] = {
		{6, 5, 10},
		{6, 5, 10},
		{6, 5, 10}
	};

	int result[3][3] = {};
	


	addMatricesWithCuda(a, b, result);

}

void addMatricesWithCuda(const int a[][3], const int b[][3], int c[][3])
{
	cudaError_t cudaStatus;
	int *pA = 0;
	int *pB = 0;
	int *pC = 0;

	//1. Setup device.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		cleanUpMatrixOperation(pA, pB, pC);
	}

	//2. Allocate memory for 3 matrices
	int sizeOfMatrices = (3 * sizeof(int)) * 3;

	cudaStatus = cudaMalloc((void**)&pA, sizeOfMatrices);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed allocating mem for matrix");
		cleanUpMatrixOperation(pA, pB, pC);
	}

	cudaStatus = cudaMalloc((void**)&pB, sizeOfMatrices);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed allocating mem for matrix");
		cleanUpMatrixOperation(pA, pB, pC);
	}

	cudaStatus = cudaMalloc((void**)&pC, sizeOfMatrices);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed allocating mem for matrix");
		cleanUpMatrixOperation(pA, pB, pC);
	}

	//3. Copy memory from host structures to device.
	cudaStatus = cudaMemcpy(pA, a, sizeOfMatrices, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed copying memory for matrix");
		cleanUpMatrixOperation(pA, pB, pC);
	}

	cudaStatus = cudaMemcpy(pB, b, sizeOfMatrices, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed copying memory for matrix");
		cleanUpMatrixOperation(pA, pB, pC);
	}

	cudaStatus = cudaMemcpy(pC, c, sizeOfMatrices, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed copying memory for matrix");
		cleanUpMatrixOperation(pA, pB, pC);
	}

	addMarticesKernel<<<1, dim3(3, 3) >>>((int**)pC, (const int**)&pA, (const int**)&pB);

	// 4. Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "add matrix launch failed: %s\n", cudaGetErrorString(cudaStatus));
		cleanUpMatrixOperation(pA, pB, pC);
	}

	// 5. cudaDeviceSynchronize waits for the kernel to finish, and returns
	//	  any errors encountered during the launch?
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addMatrix!\n", cudaStatus);
		cleanUpMatrixOperation(pA, pB, pC);
	}

	
	cleanUpMatrixOperation(pA, pB, pC);
}

void cleanUpMatrixOperation(int *pA, int *pB, int *pResult)
{
	cudaFree(pA);
	cudaFree(pB);
	cudaFree(pResult);
}

int main()
{
	cudaMatrixExample();
    return 0;
}

void printArray(const int a[], int size)
{
	printf("{");
	for (int i = 0; i < size; i++)
	{
		if (i % 50 == 0 && i != 0) {
			printf("\n");
		}
		printf("%d", a[i]);
		if (i != size -1)
		{
			printf(",");
		}
	}
	printf("}");
}

void fillArray(int a[], int size) 
{
	for (int i = 0; i < size; i += 2)
	{
		a[i] = 1;
		a[i + 1] = 2;
	}
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size, unsigned int *blocks, unsigned int *threadsPerBlock)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	 	
	
	

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<5, 10>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
