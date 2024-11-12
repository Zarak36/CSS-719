#include <iostream>
#include <vector>
#include <cmath>
#include <pthread.h>
#include <mutex>
#include <chrono>

const int LIMIT = 1000000;  // Upper limit for the sieve (large enough to get 1000 primes)
std::vector<bool> is_prime(LIMIT, true);
std::vector<int> primes;
pthread_mutex_t primes_mutex;

struct ThreadData {
    int start;
    int end;
};

void* sieve_segment(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int start = data->start;
    int end = data->end;
    for (int i = start; i <= end; ++i) {
        if (is_prime[i]) {
            for (int j = i * i; j < LIMIT; j += i) {
                is_prime[j] = false;
            }
        }
    }
    pthread_exit(nullptr);
}

void* collect_primes(void* arg) {
    for (int i = 2; i < LIMIT && primes.size() < 1000; ++i) {
        if (is_prime[i]) {
            pthread_mutex_lock(&primes_mutex);
            if (primes.size() < 1000)  // Check again after acquiring the lock
                primes.push_back(i);
            pthread_mutex_unlock(&primes_mutex);
        }
    }
    pthread_exit(nullptr);
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();

    int num_threads = 4;  // For example, you can adjust this as needed
    int segment_size = std::sqrt(LIMIT) / num_threads;
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];

    // Initialize the mutex
    pthread_mutex_init(&primes_mutex, nullptr);

    // Launch threads for parallel sieving
    for (int i = 0; i < num_threads; ++i) {
        thread_data[i].start = 2 + i * segment_size;
        thread_data[i].end = std::min(thread_data[i].start + segment_size - 1, (int)std::sqrt(LIMIT));
        pthread_create(&threads[i], nullptr, sieve_segment, &thread_data[i]);
    }

    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }

    // Collect the first 1000 primes using a separate thread
    pthread_t collect_thread;
    pthread_create(&collect_thread, nullptr, collect_primes, nullptr);
    pthread_join(collect_thread, nullptr);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;

    // Destroy the mutex
    pthread_mutex_destroy(&primes_mutex);

    // Print the first 1000 primes
    for (size_t i = 0; i < primes.size(); ++i) {
        std::cout << primes[i] << " ";
        if ((i + 1) % 10 == 0) std::cout << std::endl;
    }

    std::cout << "\nTime taken: " << elapsed_time.count() << " seconds" << std::endl;

    return 0;
}