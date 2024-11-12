#include <iostream>
#include <vector>
#include <limits>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

const int INF = std::numeric_limits<int>::max();

// Functor for Floyd-Warshall step in Thrust
struct FloydWarshallStep {
    int* dist;
    int n;
    int k;

    FloydWarshallStep(int* dist, int n, int k) : dist(dist), n(n), k(k) {}

    __device__ void operator()(int idx) {
        int i = idx / n;
        int j = idx % n;
        int ik = i * n + k;
        int kj = k * n + j;

        if (dist[ik] != INF && dist[kj] != INF && dist[ik] + dist[kj] < dist[idx]) {
            dist[idx] = dist[ik] + dist[kj];
        }
    }
};

void floydWarshallThrust(std::vector<std::vector<int>>& dist, int n) {
    // Flatten the matrix into a 1D array for GPU processing
    std::vector<int> flat_dist(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            flat_dist[i * n + j] = dist[i][j];
        }
    }

    // Copy data to the GPU using Thrust's device vector
    thrust::device_vector<int> d_dist = flat_dist;

    // Start measuring time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Run Floyd-Warshall for each intermediate vertex k
    for (int k = 0; k < n; ++k) {
        FloydWarshallStep fwStep(thrust::raw_pointer_cast(d_dist.data()), n, k);
        thrust::for_each(thrust::counting_iterator<int>(0),
                         thrust::counting_iterator<int>(n * n),
                         fwStep);
    }

    // Stop measuring time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy results back to host
    thrust::copy(d_dist.begin(), d_dist.end(), flat_dist.begin());

    // Rebuild the 2D matrix from the flattened array
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            dist[i][j] = flat_dist[i * n + j];
        }
    }

    // Clean up events
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

    // Run Floyd-Warshall algorithm using Thrust
    floydWarshallThrust(dist, n);

    return 0;
}