#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <mutex>
#include "SGD.h"
#include "helper.h"

void stochastic_gradient_descent_without_multithreading(std::vector<double>& X, std::vector<double>& Y) {
    double alpha = 0.0001;  // Learning rate
    int iterations = 150000;
    double theta0 = 0.0;
    double theta1 = 0.0;
    int m = X.size();  // Number of training examples

    for (int iter = 0; iter < iterations; ++iter) {
        // Loop over each training example
        for (int i = 0; i < m; ++i) {
            // Calculate the hypothesis and the error
            double error = hypothesis(theta0, theta1, X[i]) - Y[i];

            // Update the parameters for each training example
            theta0 -= alpha * error;
            theta1 -= alpha * error * X[i];
        }

        // Uncomment to print cost every 1000 iterations (optional)
        // if (iter % 1000 == 0) {
        //     std::cout << "Iteration " << iter << ": Cost = " << computeCost(X, Y, theta0, theta1) << std::endl;
        // }
    }
    std::cout << ": Cost = " << computeCost(X, Y, theta0, theta1) << std::endl;
    std::cout << "Using Stochastic Gradient Descent without multithreading:" << std::endl;
    std::cout << "Optimized Theta0: " << theta0 << std::endl;
    std::cout << "Optimized Theta1: " << theta1 << std::endl;
}

// Function to compute and update the gradients for a subset of the data
void stochasticComputeGradient(const std::vector<double>& X, const std::vector<double>& Y, 
                               double& theta0, double& theta1, int start, int end, double alpha, std::mutex& mtx) {
    for (int i = start; i < end; ++i) {
        double prediction = hypothesis(theta0, theta1, X[i]);
        double error = prediction - Y[i];

        // Lock the mutex to ensure thread-safe parameter updates
        std::lock_guard<std::mutex> lock(mtx);

        // Update the parameters for this subset of data
        theta0 -= alpha * error;
        theta1 -= alpha * error * X[i];
    }
}

// Function to perform stochastic gradient descent with multi-threading
void stochastic_gradient_descent_with_multithreading(std::vector<double>& X, std::vector<double>& Y) {
    double theta0 = 0.0;
    double theta1 = 0.0;
    double alpha = 0.0001;  // Learning rate
    int iterations = 10000;
    int m = X.size();
    int num_threads = 4;  // Number of threads
    int chunk_size = m / num_threads;
    std::mutex mtx;  // Mutex for thread-safe parameter updates

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<std::thread> threads;

        // Split the data across threads and start them
        for (int i = 0; i < num_threads; ++i) {
            int start = i * chunk_size;
            int end = (i == num_threads - 1) ? m : start + chunk_size;
            threads.push_back(std::thread(stochasticComputeGradient, std::cref(X), std::cref(Y), std::ref(theta0), std::ref(theta1), start, end, alpha, std::ref(mtx)));
        }

        // Wait for all threads to finish
        for (auto& th : threads) {
            if (th.joinable()) {
                th.join();
            }
        }

        // Uncomment to print cost every 1000 iterations (optional)
        // if (iter % 1000 == 0) {
        //     std::cout << "Iteration " << iter << ": Cost = " << computeCost(X, Y, theta0, theta1) << std::endl;
        // }
    }
    std::cout << "Cost = " << computeCost(X, Y, theta0, theta1) << std::endl;
    std::cout << "Using Stochastic Gradient Descent with multithreading" << std::endl;
    std::cout << "Optimized Theta0: " << theta0 << std::endl;
    std::cout << "Optimized Theta1: " << theta1 << std::endl;
}

