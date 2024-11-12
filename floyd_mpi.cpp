#include <iostream>
#include <vector>
#include <mpi.h>
#include <limits>

const int INF = std::numeric_limits<int>::max();

void floydWarshallMPI(std::vector<std::vector<int>>& dist, int n, int rank, int size) {
    for (int k = 0; k < n; ++k) {
        // Broadcast the k-th row to all processes
        MPI_Bcast(dist[k].data(), n, MPI_INT, k / (n / size), MPI_COMM_WORLD);

        // Each process updates its part of the matrix
        for (int i = rank * (n / size); i < (rank + 1) * (n / size) && i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (dist[i][k] != INF && dist[k][j] != INF && dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }

    // Gather the updated matrix rows from each process
    if (rank == 0) {
        // Root process collects rows from all processes, including itself
        for (int i = 1; i < size; ++i) {
            int start_row = i * (n / size);
            int end_row = std::min(start_row + (n / size), n);

            for (int row = start_row; row < end_row; ++row) {
                MPI_Recv(dist[row].data(), n, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        // Each non-root process sends its rows to process 0
        int start_row = rank * (n / size);
        int end_row = std::min(start_row + (n / size), n);

        for (int row = start_row; row < end_row; ++row) {
            MPI_Send(dist[row].data(), n, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 4; // Number of vertices in the graph (example)
    std::vector<std::vector<int>> dist(n, std::vector<int>(n, INF));

    // Initialize the graph on process 0
    if (rank == 0) {
        dist = {
            {0, 3, INF, 5},
            {2, 0, INF, 4},
            {INF, 1, 0, INF},
            {INF, INF, 2, 0}
        };
    }

    // Broadcast the initial graph to all processes
    for (int i = 0; i < n; ++i) {
        MPI_Bcast(dist[i].data(), n, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Start measuring time
    double start_time = MPI_Wtime();

    // Run Floyd-Warshall algorithm
    floydWarshallMPI(dist, n, rank, size);

    // End measuring time
    double end_time = MPI_Wtime();

    // Print the shortest path matrix on process 0
    if (rank == 0) {
        std::cout << "Shortest distances between every pair of vertices:\n";
        for (const auto& row : dist) {
            for (int val : row) {
                if (val == INF) std::cout << "INF ";
                else std::cout << val << " ";
            }
            std::cout << "\n";
        }
        // Print the duration in seconds
        std::cout << "Time taken: " << (end_time - start_time) << " seconds\n";
    }

    MPI_Finalize();
    return 0;
}