# Core Concepts

## 2D block of threads in a single block

The code below defines a launch config of 1 block set up with 10 threads across (x) and 5 threads down (y) marking a total of 5 * 10 = 50.

```c++
    addKernel<<<1, dim3(10, 5) >>>(dev_c, dev_a, dev_b);
```

The code below shows how obtain a unique index for this setup

```c++
__global__
void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x + (threadIdx.y * blockDim.x);

    c[i] = a[i] + b[i];
}
```

```blockDim.x``` allows access to the numbers of threads in block.
```threadIdx.x``` refers to the horizontal index of threads.
```threadIdx.y``` refers to the vertical index of threads.

## Multiple blocks using a single row of threads

The launch configuration below defines launching 5 blocks with 10 threads (x) with a total number of threads again being 50.

```c++
addKernel<<<5, 10>>>(dev_c, dev_a, dev_b);
```

The function below shows how to get a unique id for each of the index using the configuration above.

```c++
__global__ 
void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	
    c[i] = a[i] + b[i];
}
```
