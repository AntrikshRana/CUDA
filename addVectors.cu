/*
    Simple CUDA program to add two vectors
*/

#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

// This is the CUDA Kernel
__global__ void addVectors(int* A, int* B, int* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // Pointer declaration and memory allocation
    int n = 4;
    int* d_A;
    int* d_B;
    int* d_C;

    int* A = (int*)malloc(n*sizeof(int));
    int* B = (int*)malloc(n*sizeof(int));
    int* C = (int*)malloc(n*sizeof(int));

    // Taking Input
    std::cout<<"Input A : "; for(int i=0; i<n; i++) std::cin>>A[i];
    std::cout<<"Input B : "; for(int i=0; i<n; i++) std::cin>>B[i];
    std::cout<<'\n';

    // Allocating CUDA memory
    cudaMalloc((void**)&d_A, n*sizeof(int));
    cudaMalloc((void**)&d_B, n*sizeof(int));
    cudaMalloc((void**)&d_C, n*sizeof(int));

    // Copying data from CPU memory to GPU memory
    cudaMemcpy(d_A, A, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n*sizeof(int), cudaMemcpyHostToDevice);
    
    addVectors<<<1, 4>>>(d_A, d_B, d_C, n); // Called the Kernel

    cudaMemcpy(C, d_C, n*sizeof(int), cudaMemcpyDeviceToHost); // Storing the output

    for(int i=0; i<n; i++) std::cout<< C[i] << " ";

    // Free memory
    free(A); free(B); free(C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}