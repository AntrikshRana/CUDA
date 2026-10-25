#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

void checkCuda(cudaError_t err, const char* operation) {
    if(err != cudaSuccess) {
        std::cerr << operation << " failed!" << cudaGetErrorString(err) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

__global__ void matrixMultiplication(int *A, int *B, int *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * n)
        return;
    int row = i / n;
    int col = i % n;

    int sum = 0;

    for (int k = 0; k < n; k++)
    {
        sum += A[row * n + k] * B[k * n + col];
    }

    C[row * n + col] = sum;
}

int main()
{
    int n = 512;
    int threadsPerBlock = 128;
    int matDim = n * n;

    int *A = (int *)malloc(matDim * sizeof(int));
    int *B = (int *)malloc(matDim * sizeof(int));
    int *C = (int *)malloc(matDim * sizeof(int));
    int *res = (int *)malloc(matDim * sizeof(int));

    if (A == nullptr || B == nullptr || C == nullptr)
    {

        std::cerr << "Host memory allocation failed!";
        free(A);
        free(B);
        free(C);
        free(res);
        std::exit(EXIT_FAILURE);
    }

    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < n; col++)
        {
            A[row * n + col] = row + col;
            B[row * n + col] = 2 * (row + col);
        }
    }

    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            res[row * n + col] = 0;

            for (int k = 0; k < n; k++) {
                res[row * n + col] += A[row * n + k] * B[k * n + col];
            }
        }
    }

    int *d_A; int *d_B; int *d_C; // Device mempry allocation

    checkCuda(cudaMalloc((void **)&d_A, matDim * sizeof(int)), "Allocate A");
    checkCuda(cudaMalloc((void **)&d_B, matDim * sizeof(int)), "Allocate B");
    checkCuda(cudaMalloc((void **)&d_C, matDim * sizeof(int)), "Allocate C");

    checkCuda(cudaMemcpy(d_A, A, matDim * sizeof(int), cudaMemcpyHostToDevice), "Copy A");
    checkCuda(cudaMemcpy(d_B, B, matDim * sizeof(int), cudaMemcpyHostToDevice), "Copy B");

    int blocksPerGrid = (matDim + threadsPerBlock - 1) / threadsPerBlock;
    matrixMultiplication<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);

    checkCuda(cudaDeviceSynchronize(), "Synchronize");

    checkCuda(cudaMemcpy(C, d_C, matDim * sizeof(int), cudaMemcpyDeviceToHost), "Copy C");

    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < n; col++)
        {
            if (C[row * n + col] != res[row * n + col])
            {
                std::cout << "Unexpected output!\n";
                std::exit(EXIT_FAILURE);
            }
        }
    }

    std::cout << "Kernel passed!\n";

    // Free the allocated memory
    free(A);
    free(B);
    free(C);
    free(res);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}