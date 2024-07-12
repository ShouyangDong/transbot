#include <stdio.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda.h>
__global__ void add(float *output, float *input1, float *input2)
{
  if (blockIdx.x < 18)
  {
    {
      if (threadIdx.x < 1024)
      {
        {
          int index = (blockIdx.x * 1024) + threadIdx.x;
          output[index] = input1[index] + input2[index];
        }
      }
    }
  }
}

extern "C" void add_kernel(float *C, float *A, float *B, int size) {
            float *d_A, *d_B, *d_C;

            cudaMalloc(&d_A, size * sizeof(float));
            cudaMalloc(&d_B, size * sizeof(float));
            cudaMalloc(&d_C, size * sizeof(float));

            cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, B, size * sizeof(float), cudaMemcpyHostToDevice);

            dim3 blockSize(1024);
            dim3 numBlocks(256);

            add<<<numBlocks, blockSize>>>(d_C, d_A, d_B);

            cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            }
        