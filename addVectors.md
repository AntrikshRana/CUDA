# Vector addition

This program benchmarks the time taken to execute a kernel 1000 times

The benchmark is performed on two vectors `A[1000000] and B[1000000]` the result is then stored in the third vector `C[1000000]` and then the output is confirmed by the cpu.

After testing on different `threadsPerBlock` & `blocksPerGrid` these are the results.

|Serial|BLOCKS|THREADS|AVERAGE TIME|
|------|------|-------|------------|
|1     |7813  |128    |0.117469 ms |
|2     |3907  |256    |0.117319 ms |
|3     |1954  |512    |0.117379 ms |
|4     |977   |1024   |0.116698 ms |

Even though dimensions are different the average time is almost similar. From this we can conclude that the arithmetic operations (IN THIS CASE) are not the limiting factor.

They could be if we did 100 arithmetic per element...That brings is to 10^11 operations, compared to that - copying, reading, and writing may not affect the output (IN THIS CASE)

## What limits the speed
The actual limiting factor is the memory read write not the addition part

### What I learned
If we are performing operations but the result does not affect the result then the compiler can perform `Dead-Code Elimination`. The compiler will just NOT perform those operations because they have no impact.

Various methods like
- cudaMalloc()
- cudaMemcpy()
- cudaFree()
- cudaEventCreate()
- cudaGetLastError()
- cudaDeviceSynchronize()
- and much more..

I'm not writing parameters because I dont want to!

# THANK YOU FOR READING