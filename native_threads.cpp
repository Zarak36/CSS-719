#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include <mutex>
#include <chrono>

const int LIMIT = 1000000;  // Upper limit for the sieve (large enough to get 1000 primes)
std::vector<bool> is_prime(LIMIT, true);
std::vector<int> primes;
std::mutex primes_mutex;

void sieve_segment(int start, int end) {
    for (int i = start; i <= end; ++i) {
        if (is_prime[i]) {
            for (int j = i * i; j < LIMIT; j += i) {
                is_prime[j] = false;
            }
        }
    }
}

void collect_primes() {
    for (int i = 2; i < LIMIT && primes.size() < 1000; ++i) {
        if (is_prime[i]) {
            std::lock_guard<std::mutex> guard(primes_mutex);
            if (primes.size() < 1000)  // Check again after acquiring the lock
                primes.push_back(i);
        }
    }
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();

    int num_threads = std::thread::hardware_concurrency();
    int segment_size = std::sqrt(LIMIT) / num_threads;
    std::vector<std::thread> threads;

    // Launch threads for parallel sieving
    for (int i = 0; i < num_threads; ++i) {
        int start = 2 + i * segment_size;
        int end = std::min(start + segment_size - 1, (int)std::sqrt(LIMIT));
        threads.emplace_back(sieve_segment, start, end);
    }

    for (auto &t : threads) {
        t.join();
    }

    // Clear thread list and launch threads to collect the first 1000 primes
    threads.clear();
    threads.emplace_back(collect_primes);
    for (auto &t : threads) {
        t.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;

    // Print the first 1000 primes
    for (size_t i = 0; i < primes.size(); ++i) {
        std::cout << primes[i] << " ";
        if ((i + 1) % 10 == 0) std::cout << std::endl;
    }

    std::cout << "\nTime taken: " << elapsed_time.count() << " seconds" << std::endl;

    return 0;
}