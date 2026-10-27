/*
    Tiled matrix multiplication
*/

#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#define TILE_SIZE 16

void checkCuda(cudaError_t err, const char* operation) {
    if(err != cudaSuccess) {
        std::cerr << operation << " failed!" << cudaGetErrorString(err) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

__global__ void matrixMultiplication(const int *A,const int *B, int *C, int n)
{
    __shared__ int tileA[TILE_SIZE][TILE_SIZE];
    __shared__ int tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y; // Global output coordinates
    int col = blockIdx.x * TILE_SIZE + threadIdx.x; // Global output coordinates

    int sum = 0;

    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE; // Iterate over tiles along the multiplication dimension

    for(int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        // Load tiles ehlie handling boundaries
        tileA[threadIdx.y][threadIdx.x] = (row < n && aCol < n) ? A[row * n + aCol] : 0;
        tileB[threadIdx.y][threadIdx.x] = (bRow < n && col < n) ? B[bRow * n + col] : 0;
        
        __syncthreads(); // Wait until the entire tiles are loaded

        for(int k=0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads(); // Ensure all threads finish using the tiles before overriding them.
    }

    if(row < n && col < n) {
        C[row * n + col] = sum;
    }
}

int main()
{
    int n = 512;
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
    
    cudaEvent_t start, stop;
    float milliseconds = 0.0f;

    checkCuda(cudaEventCreate(&start), "Create start event");
    checkCuda(cudaEventCreate(&stop), "Create stop event");

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    // Warm-up: run the kernel 10 times;
    for(int i=0; i<10; i++) {
        matrixMultiplication<<<blocks, threads>>>(d_A, d_B, d_C, n);
    }
    
    checkCuda(cudaGetLastError(), "Warm-up launch");
    checkCuda(cudaDeviceSynchronize(), "Warm-up execute");
    
    int iterations = 10;
    
    checkCuda(cudaEventRecord(start), "Record start");
    
    for(int i=0; i<iterations; i++) {
        matrixMultiplication<<<blocks, threads>>>(d_A, d_B, d_C, n);
    }

    checkCuda(cudaGetLastError(), "Benchmark Launch");
    checkCuda(cudaEventRecord(stop), "Record stop");
    checkCuda(cudaEventSynchronize(stop), "Wait for benchmark");

    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "Calculate elapsed time");
    std::cout << "-----------------------------------------------------\n";
    std::cout << "Total elapsed time: " << milliseconds << "ms\n";
    std::cout << "Average kernel time: " << milliseconds / iterations << "ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    checkCuda(cudaMemcpy(C, d_C, matDim * sizeof(int), cudaMemcpyDeviceToHost), "Copy C");


    for(int row=0; row<n; row++) {
        for(int col=0; col<n; col++) {
            if(res[row * n + col] != C[row * n + col]) {
                std::cerr << "Mismatch at index: " << row * n + col
                          << "Expected value: " << res[row * n + col]
                          << "Actual: " << C[row * n + col] << "\n";
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