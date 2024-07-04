# Transcompile c code to CUDA Code

orginal c code:
```
void add_kernel(float* output, float* input1, float* input2) {
    for (int i = 0; i < 18; i++) {
        for (int j = 0; j < 128; j++) {
            int index = i * 128 + j;
            output[index] = input1[index] + input2[index];
        }
    }
}
```

action 1：
add CUDA function prefix: 

```
__global__ void add_kernel(float* output, float* input1, float* input2) {
    for (int i = 0; i < 18; i++) {
        for (int j = 0; j < 128; j++) {
            int index = i * 128 + j;
            output[index] = input1[index] + input2[index];
        }
    }
}
```


action 2:

loop fusion
```
__global__ void add_kernel(float* output, float* input1, float* input2) {
    for (int index = 0; index < 2304; index++) {
            output[index] = input1[index] + input2[index];
    }
}
```

action 3:

loop split
```
__global__ void add_kernel(float* output, float* input1, float* input2) {
    for (int i_o = 0; i_o < 3) {
        for (int i_in = 0; i_in < 1024; i_in++) {
            if (i_o * 1024 + i_in < 2304) {
                output[i_o * 1024 + i_in] = input1[i_o * 1024 + i_in] + input2[i_o * 1024 + i_in];
            }
        }
    }
}
```

action 4:
thread bind
```
__global__ void add_kernel(float* output, float* input1, float* input2) {
    if (blockIdx.x < 3) {
        if (threadIdx.x < 1024) {
            if (blockIdx.x * 1024 + threadIdx.x < 2304) {
                output[blockIdx.x * 1024 + threadIdx.x] = input1[blockIdx.x * 1024 + threadIdx.x] + input2[blockIdx.x * 1024 + threadIdx.x];
            }
        }
    }
}
```

## Finally we can get the CUDA C kernel
```
__global__ void add_kernel(float* output, float* input1, float* input2) {
    if (blockIdx.x < 3) {
        if (threadIdx.x < 1024) {
            if (blockIdx.x * 1024 + threadIdx.x < 2304) {
                output[blockIdx.x * 1024 + threadIdx.x] = input1[blockIdx.x * 1024 + threadIdx.x] + input2[blockIdx.x * 1024 + threadIdx.x];
            }
        }
    }
}
```
