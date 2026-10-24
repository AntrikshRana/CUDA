/*
    Simple CUDA program to add two vectors
*/

#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

// Check for errors and execute when somethgins is wrong
void checkCuda(cudaError_t err, const char *operation)
{
    if (err != cudaSuccess)
    {
        std::cerr << operation << "failed: " << cudaGetErrorString(err) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

// This is the CUDA Kernel
__global__ void addVectors(int *A, int *B, int *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    // Pointer declaration and memory allocation
    int n = 1000000;

    int *A = (int *)malloc(n * sizeof(int));
    int *B = (int *)malloc(n * sizeof(int));
    int *C = (int *)malloc(n * sizeof(int));
    int *res = (int *)malloc(n * sizeof(int)); // Result vector to verify the kernel's output

    if (A == nullptr || B == nullptr || C == nullptr || res == nullptr)
    {
        std::cerr << "Host memory allocation failed\n";
        free(A);
        free(B);
        free(C);
        free(res);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < n; i++)
    {
        A[i] = i;
        B[i] = i * 2;
        res[i] = i * 3;
    }

    int *d_A;
    int *d_B;
    int *d_C;

    checkCuda(cudaMalloc((void **)&d_A, n * sizeof(int)), "cudaMalloc d_A");
    checkCuda(cudaMalloc((void **)&d_B, n * sizeof(int)), "cudaMalloc d_B");

    cudaError_t err = cudaMalloc((void **)&d_C, n * sizeof(int));
    if (err != cudaSuccess)
    {
        cudaFree(d_A);
        cudaFree(d_B);
        free(A);
        free(B);
        free(C);
        free(res);
        checkCuda(err, "cudaMalloc d_C");
    }

    // Copying data from CPU memory to GPU memory
    checkCuda(cudaMemcpy(d_A, A, n * sizeof(int), cudaMemcpyHostToDevice), "Copy A");
    checkCuda(cudaMemcpy(d_B, B, n * sizeof(int), cudaMemcpyHostToDevice), "Copy B");

    for (int threadsPerBlock = 128; threadsPerBlock <= 1024; threadsPerBlock *= 2)
    {
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        cudaEvent_t start, stop;
        float milliseconds = 0.0f;

        checkCuda(cudaEventCreate(&start), "Create start event");
        checkCuda(cudaEventCreate(&stop), "Create stop event");

        // Warm-up: run the kernel 10 times
        for (int i = 0; i < 10; i++)
        {
            addVectors<<<blocksPerGrid, threadsPerBlock>>>(
                d_A, d_B, d_C, n);
        }

        checkCuda(cudaGetLastError(), "Warm-up launch");
        checkCuda(cudaDeviceSynchronize(), "Warm-up execution");

        // Benchmark 1000 kernel launches
        int iterations = 1000;

        checkCuda(cudaEventRecord(start), "Record start");

        for (int i = 0; i < iterations; i++)
        {
            addVectors<<<blocksPerGrid, threadsPerBlock>>>(
                d_A, d_B, d_C, n);
        }

        checkCuda(cudaGetLastError(), "Benchmark launch");
        checkCuda(cudaEventRecord(stop), "Record stop");
        checkCuda(cudaEventSynchronize(stop), "Wait for benchmark");

        checkCuda(
            cudaEventElapsedTime(&milliseconds, start, stop),
            "Calculate elapsed time");
        std::cout << "--------------------------------------------------\n";
        std::cout << "BLOCKS: " << blocksPerGrid << " THREADS: " << threadsPerBlock << "\n";
        std::cout << "Total time: " << milliseconds << " ms\n";
        std::cout << "Average kernel time: "
                  << milliseconds / iterations << " ms\n";

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        checkCuda(cudaMemcpy(C, d_C, n * sizeof(int), cudaMemcpyDeviceToHost), "Copy C");

        for (int i = 0; i < n; i++)
        {
            if (C[i] != res[i])
            {
                std::cerr << "Mismatch at index: " << i
                          << "Expected value: " << res[i]
                          << "Actual: " << C[i] << "\n";
                std::exit(EXIT_FAILURE);
            }
        }

        std::cout << "All results verified successfully\n";
    }

    // Free memory
    free(A);
    free(B);
    free(C);
    free(res);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}