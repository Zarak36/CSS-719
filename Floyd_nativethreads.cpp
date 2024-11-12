#include <iostream>
#include <vector>
#include <thread>
#include <limits>

// Define the number of threads
const int NUM_THREADS = 4;
const int INF = std::numeric_limits<int>::max();

void floydWarshallThreaded(std::vector<std::vector<int>>& dist, int k, int start, int end) {
    int n = dist.size();
    for (int i = start; i < end; ++i) {
        for (int j = 0; j < n; ++j) {
            if (dist[i][k] != INF && dist[k][j] != INF && dist[i][k] + dist[k][j] < dist[i][j]) {
                dist[i][j] = dist[i][k] + dist[k][j];
            }
        }
    }
}

void floydWarshall(std::vector<std::vector<int>>& dist) {
    int n = dist.size();
   
    for (int k = 0; k < n; ++k) {
        // Create a vector to hold the threads
        std::vector<std::thread> threads;
        int chunk_size = (n + NUM_THREADS - 1) / NUM_THREADS;  // Calculate row chunk size for each thread

        // Launch threads
        for (int t = 0; t < NUM_THREADS; ++t) {
            int start = t * chunk_size;
            int end = std::min(start + chunk_size, n);  // Ensure the end does not exceed n
            threads.emplace_back(floydWarshallThreaded, std::ref(dist), k, start, end);
        }

        // Join all threads
        for (auto& th : threads) {
            th.join();
        }
    }
}

int main() {
    // Initialize the graph with some sample values
    int n = 4;  // Number of vertices
    std::vector<std::vector<int>> dist = {
        {0, 3, INF, 5},
        {2, 0, INF, 4},
        {INF, 1, 0, INF},
        {INF, INF, 2, 0}
    };

    // Run Floyd-Warshall algorithm with threading
    floydWarshall(dist);

    // Print the shortest path distances
    std::cout << "Shortest distances between every pair of vertices:\n";
    for (const auto& row : dist) {
        for (int val : row) {
            if (val == INF) std::cout << "INF ";
            else std::cout << val << " ";
        }
        std::cout << "\n";
    }

    return 0;
}