#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <iostream>
#include <chrono>

const int LIMIT = 100000000;  // Upper limit for the sieve
const int TARGET_PRIMES = 1000;  // Number of primes we want to find

// Custom functor to check if a number is prime
struct is_prime
{
    __host__ __device__
    bool operator()(const int x)
    {
        if (x < 2) return false;
        if (x == 2) return true;
        if (x % 2 == 0) return false;
       
        for (int i = 3; i * i <= x; i += 2)
        {
            if (x % i == 0) return false;
        }
        return true;
    }
};

// Functor to mark non-prime numbers in the sieve
struct mark_non_primes
{
    __host__ __device__
    void operator()(int& x)
    {
        if (x < 2)
        {
            x = 0;
            return;
        }
       
        for (int i = 2; i * i <= x; i++)
        {
            if (x % i == 0 && x != i)
            {
                x = 0;
                return;
            }
        }
    }
};

int main()
{
    auto start_time = std::chrono::high_resolution_clock::now();

    // Create a sequence of numbers from 0 to LIMIT-1
    thrust::device_vector<int> numbers(LIMIT);
    thrust::sequence(numbers.begin(), numbers.end());

    // Apply the sieve operation in parallel on GPU
    thrust::for_each(thrust::device, numbers.begin(), numbers.end(), mark_non_primes());

    // Remove all non-prime numbers (marked as 0)
    thrust::device_vector<int> primes(LIMIT);
    auto new_end = thrust::remove_copy(thrust::device,
                                     numbers.begin(), numbers.end(),
                                     primes.begin(),
                                     0);

    // Resize to actual number of primes found
    primes.resize(new_end - primes.begin());

    // Copy results back to host
    thrust::host_vector<int> host_primes = primes;

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;

    // Print the first 1000 primes
    int count = 0;
    for (size_t i = 0; i < std::min(size_t(TARGET_PRIMES), host_primes.size()); ++i) {
        std::cout << host_primes[i] << " ";
        if (++count % 10 == 0) std::cout << std::endl;
    }

    std::cout << "\nFound " << host_primes.size() << " prime numbers" << std::endl;
    std::cout << "Time taken: " << elapsed_time.count() << " seconds" << std::endl;

    return 0;
}