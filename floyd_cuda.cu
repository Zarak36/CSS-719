#include <iostream>
#include <vector>
#include <limits>
#include <cuda_runtime.h>

const int INF = std::numeric_limits<int>::max();

// CUDA kernel for Floyd-Warshall algorithm
__global__ void floydWarshallKernel(int* dist, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < n) {
        int idx = i * n + j;
        int k_idx = k * n + j;
        int i_k_idx = i * n + k;
       
        if (dist[i_k_idx] != INF && dist[k_idx] != INF && dist[i_k_idx] + dist[k_idx] < dist[idx]) {
            dist[idx] = dist[i_k_idx] + dist[k_idx];
        }
    }
}

void floydWarshallCUDA(std::vector<std::vector<int>>& dist, int n) {
    int* d_dist;
    size_t size = n * n * sizeof(int);

    // Flatten the matrix into a 1D array for GPU
    std::vector<int> flat_dist(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            flat_dist[i * n + j] = dist[i][j];
        }
    }

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_dist, size);
    cudaMemcpy(d_dist, flat_dist.data(), size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + 15) / 16, (n + 15) / 16);

    // Start measuring time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Run the Floyd-Warshall algorithm on the GPU
    for (int k = 0; k < n; ++k) {
        floydWarshallKernel<<<numBlocks, threadsPerBlock>>>(d_dist, n, k);
        cudaDeviceSynchronize();
    }

    // Stop measuring time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy the result back to the host
    cudaMemcpy(flat_dist.data(), d_dist, size, cudaMemcpyDeviceToHost);

    // Copy the result back into the 2D matrix
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            dist[i][j] = flat_dist[i * n + j];
        }
    }

    // Clean up
    cudaFree(d_dist);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Output the result
    std::cout << "Shortest distances between every pair of vertices:\n";
    for (const auto& row : dist) {
        for (int val : row) {
            if (val == INF) std::cout << "INF ";
            else std::cout << val << " ";
        }
        std::cout << "\n";
    }

    std::cout << "Time taken: " << milliseconds / 1000.0 << " seconds\n";
}

int main() {
    int n = 4; // Number of vertices in the graph (example)
    std::vector<std::vector<int>> dist = {
        {0, 3, INF, 5},
        {2, 0, INF, 4},
        {INF, 1, 0, INF},
        {INF, INF, 2, 0}
    };

    // Run Floyd-Warshall algorithm using CUDA
    floydWarshallCUDA(dist, n);

    return 0;
}