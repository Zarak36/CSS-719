#include <iostream>
#include <vector>
#include <cmath>
#include <cuda.h>
#include <chrono>

const int LIMIT = 1000000;  // A large limit to ensure we find at least 1000 primes

__global__ void sieveKernel(bool* is_prime, int sqrt_limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + 2;  // Start from 2
    if (idx <= sqrt_limit && is_prime[idx]) {
        for (int j = idx * idx; j < LIMIT; j += idx) {
            is_prime[j] = false;
        }
    }
}

int main() {
    // CPU start time
    auto cpu_start = std::chrono::high_resolution_clock::now();

    // Allocate memory on host
    bool* is_prime_h = new bool[LIMIT];
    std::fill_n(is_prime_h, LIMIT, true);
    is_prime_h[0] = is_prime_h[1] = false;  // 0 and 1 are not prime numbers

    // Allocate memory on device
    bool* is_prime_d;
    cudaMalloc(&is_prime_d, LIMIT * sizeof(bool));
    cudaMemcpy(is_prime_d, is_prime_h, LIMIT * sizeof(bool), cudaMemcpyHostToDevice);

    // CUDA event to measure GPU time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    int sqrt_limit = static_cast<int>(std::sqrt(LIMIT));
    int threads_per_block = 256;
    int blocks = (LIMIT + threads_per_block - 1) / threads_per_block;
    sieveKernel<<<blocks, threads_per_block>>>(is_prime_d, sqrt_limit);
    cudaDeviceSynchronize();

    // Stop GPU time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_milliseconds = 0;
    cudaEventElapsedTime(&gpu_milliseconds, start, stop);

    // Copy results back to host
    cudaMemcpy(is_prime_h, is_prime_d, LIMIT * sizeof(bool), cudaMemcpyDeviceToHost);

    // CPU end time
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;

    // Collect the first 1000 primes
    std::vector<int> primes;
    for (int i = 2; i < LIMIT && primes.size() < 1000; ++i) {
        if (is_prime_h[i]) {
            primes.push_back(i);
        }
    }

    // Output the first 1000 primes
    std::cout << "First 1000 primes:\n";
    for (size_t i = 0; i < primes.size(); ++i) {
        std::cout << primes[i] << " ";
        if ((i + 1) % 10 == 0) std::cout << "\n";
    }
    std::cout << "\n\n";

    // Output timing information
    std::cout << "Time taken on GPU: " << gpu_milliseconds / 1000.0 << " seconds\n";
    std::cout << "Time taken on CPU: " << cpu_duration.count() << " seconds\n";

    // Cleanup
    delete[] is_prime_h;
    cudaFree(is_prime_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}