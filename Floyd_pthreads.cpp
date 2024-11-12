#include <iostream>
#include <vector>
#include <pthread.h>
#include <limits>
#include <chrono>

const int INF = std::numeric_limits<int>::max();
const int NUM_THREADS = 4;

struct ThreadData {
    std::vector<std::vector<int>>* dist;
    int k;
    int start;
    int end;
};

void* floydWarshallThreaded(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);
    auto& dist = *(data->dist);
    int k = data->k;
    int n = dist.size();

    for (int i = data->start; i < data->end; ++i) {
        for (int j = 0; j < n; ++j) {
            if (dist[i][k] != INF && dist[k][j] != INF && dist[i][k] + dist[k][j] < dist[i][j]) {
                dist[i][j] = dist[i][k] + dist[k][j];
            }
        }
    }
    return nullptr;
}

void floydWarshall(std::vector<std::vector<int>>& dist) {
    int n = dist.size();
   
    for (int k = 0; k < n; ++k) {
        pthread_t threads[NUM_THREADS];
        ThreadData thread_data[NUM_THREADS];
        int chunk_size = (n + NUM_THREADS - 1) / NUM_THREADS;

        for (int t = 0; t < NUM_THREADS; ++t) {
            int start = t * chunk_size;
            int end = std::min(start + chunk_size, n);

            thread_data[t] = {&dist, k, start, end};
            pthread_create(&threads[t], nullptr, floydWarshallThreaded, &thread_data[t]);
        }

        for (int t = 0; t < NUM_THREADS; ++t) {
            pthread_join(threads[t], nullptr);
        }
    }
}

int main() {
    int n = 4;
    std::vector<std::vector<int>> dist = {
        {0, 3, INF, 5},
        {2, 0, INF, 4},
        {INF, 1, 0, INF},
        {INF, INF, 2, 0}
    };

    // Start measuring time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Run Floyd-Warshall algorithm
    floydWarshall(dist);

    // End measuring time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Shortest distances between every pair of vertices:\n";
    for (const auto& row : dist) {
        for (int val : row) {
            if (val == INF) std::cout << "INF ";
            else std::cout << val << " ";
        }
        std::cout << "\n";
    }

    // Print the duration in milliseconds
    std::cout << "Time taken: " << duration.count() << " milliseconds\n";

    return 0;
}
