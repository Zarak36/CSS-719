#include <iostream>
#include <vector>
#include <limits>
#include <chrono>
#include <omp.h>

const int INF = std::numeric_limits<int>::max();

void floydWarshallOpenMP(std::vector<std::vector<int>>& dist, int n) {
    #pragma omp parallel for
    for (int k = 0; k < n; ++k) {
        // Parallelize across the rows for each intermediate vertex k
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (dist[i][k] != INF && dist[k][j] != INF && dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }
}

int main() {
    int n = 4; // Number of vertices in the graph (example)
    std::vector<std::vector<int>> dist = {
        {0, 3, INF, 5},
        {2, 0, INF, 4},
        {INF, 1, 0, INF},
        {INF, INF, 2, 0}
    };

    // Start measuring time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Run Floyd-Warshall algorithm with OpenMP parallelization
    floydWarshallOpenMP(dist, n);

    // End measuring time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    // Print the shortest path matrix
    std::cout << "Shortest distances between every pair of vertices:\n";
    for (const auto& row : dist) {
        for (int val : row) {
            if (val == INF) std::cout << "INF ";
            else std::cout << val << " ";
        }
        std::cout << "\n";
    }

    // Print the execution time
    std::cout << "Time taken: " << duration.count() << " seconds\n";

    return 0;
}