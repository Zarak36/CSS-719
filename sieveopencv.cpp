#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

const int LIMIT = 1000000;  // A limit that ensures we can find at least 1000 primes

int main() {
    // Initialize the sieve array
    std::vector<bool> is_prime(LIMIT, true);
    int sqrt_limit = std::sqrt(LIMIT);

    // Start timing
    double start_time = omp_get_wtime();

    // Mark non-prime numbers in parallel
    #pragma omp parallel for
    for (int i = 2; i <= sqrt_limit; ++i) {
        if (is_prime[i]) {
            // Mark multiples of the current prime in parallel
            #pragma omp parallel for
            for (int j = i * i; j < LIMIT; j += i) {
                is_prime[j] = false;
            }
        }
    }

    // Gather all prime numbers
    std::vector<int> primes;
    for (int i = 2; i < LIMIT; ++i) {
        if (is_prime[i]) {
            primes.push_back(i);
        }
    }

    // Stop timing
    double end_time = omp_get_wtime();
   
    // Output the first 1000 primes and the time taken
    std::cout << "First 1000 primes:\n";
    for (int i = 0; i < std::min(1000, static_cast<int>(primes.size())); ++i) {
        std::cout << primes[i] << " ";
        if ((i + 1) % 10 == 0) std::cout << std::endl;
    }
    std::cout << "\nTime taken: " << (end_time - start_time) << " seconds" << std::endl;

    return 0;
}