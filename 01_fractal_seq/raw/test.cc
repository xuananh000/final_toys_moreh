#include <hip/hip_runtime.h>
#include <iostream>

__global__ void vector_add(float *out, const float *a, const float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

int main(){
    float *a, *b, *c;
    float *d_a, *d_b, *d_c; // device pointers for a, b, c
    int N = 1<<20;

    hipMalloc(&d_a, N*sizeof(float));
    hipMalloc(&d_b, N*sizeof(float));
    hipMalloc(&d_c, N*sizeof(float));

    a = (float*)malloc(N*sizeof(float));
    b = (float*)malloc(N*sizeof(float));
    c = (float*)malloc(N*sizeof(float));

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Copy host vectors to device
    hipMemcpy(d_a, a, N*sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b, b, N*sizeof(float), hipMemcpyHostToDevice);

    hipStream_t stream0, stream1;
    hipStreamCreate(&stream0);
    hipStreamCreate(&stream1);

    // Vector addition on stream0
    hipLaunchKernelGGL(vector_add, dim3((N+255)/256), dim3(256), 0, stream0, d_c, d_a, d_b, N/2);

    // Vector addition on stream1
    // Adjust the pointers for the second half
    hipLaunchKernelGGL(vector_add, dim3((N+255)/256), dim3(256), 0, stream1, d_c + N/2, d_a + N/2, d_b + N/2, N - N/2);

    // Synchronize streams
    hipStreamSynchronize(stream0);
    hipStreamSynchronize(stream1);

    // Copy result back to host
    hipMemcpy(c, d_c, N*sizeof(float), hipMemcpyDeviceToHost);

    // Verify results
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (c[i] != a[i] + b[i]) {
            std::cerr << "Mismatch at index " << i << ": " << a[i] << " + " << b[i]
                      << " != " << c[i] << "\n";
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Vector addition successful!" << std::endl;
    }

    // Clean up
    free(a);
    free(b);
    free(c);
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    hipStreamDestroy(stream0);
    hipStreamDestroy(stream1);

    return 0;
}
