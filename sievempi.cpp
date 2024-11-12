#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>

const int LIMIT = 1000000;  // A limit that ensures we can find at least 1000 primes

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize the sieve array
    std::vector<bool> is_prime(LIMIT, true);
    int local_start, local_end;
    int global_start = 2;  // Starting point for primes
    int sqrt_limit = std::sqrt(LIMIT);

    // Determine the range for each process
    int range = (LIMIT - 2) / size;
    local_start = 2 + rank * range;  
    local_end = (rank == size - 1) ? LIMIT : local_start + range;

    // Start timing
    double start_time = MPI_Wtime();

    // Mark non-prime numbers in the local range
    for (int i = 2; i <= sqrt_limit; ++i) {
        if (is_prime[i]) {
            // Broadcast the current prime to all processes
            MPI_Bcast(&i, 1, MPI_INT, 0, MPI_COMM_WORLD);
            // Mark multiples of the current prime in the local range
            for (int j = std::max(i * i, local_start + (i - local_start % i) % i); j < local_end; j += i) {
                is_prime[j] = false;
            }
        }
    }

    // Gather all prime numbers
    std::vector<int> local_primes;
    for (int i = local_start; i < local_end; ++i) {
        if (is_prime[i]) {
            local_primes.push_back(i);
        }
    }

    // Collect results
    int local_count = local_primes.size();
    int global_count;
    MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Prepare to gather all primes at the root process
    std::vector<int> all_primes;
    if (rank == 0) {
        all_primes.resize(global_count);
    }

    // Gather all local primes to the root process
    MPI_Gather(local_primes.data(), local_count, MPI_INT,
               all_primes.data(), local_count, MPI_INT,
               0, MPI_COMM_WORLD);

    // Stop timing
    double end_time = MPI_Wtime();
   
    // Output the first 1000 primes from the root process and the time taken
    if (rank == 0) {
        std::cout << "First 1000 primes:\n";
        for (int i = 0; i < std::min(1000, global_count); ++i) {
            std::cout << all_primes[i] << " ";
            if ((i + 1) % 10 == 0) std::cout << std::endl;
        }
        std::cout << "\nTime taken: " << (end_time - start_time) << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}