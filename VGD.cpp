#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <mutex>
#include "VGD.h"
#include "helper.h"

void vanilla_gradient_descent_without_multithreading(std::vector<double>&X,std::vector<double>&Y)
{
    double alpha = 0.0001;
    int iterations = 1000000;

    //initial parameters guess
    double theta0 = 0.0;
    double theta1 = 0.0;

    int m = X.size();
    
    for(int iter = 0;iter<iterations;iter++)
    {
        double errorTheta0 = 0.0;
        double errorTheta1 = 0.0;

        // Calculate the gradient
        for (int i = 0; i < m; ++i) {
            double error = hypothesis(theta0, theta1, X[i]) - Y[i];
            errorTheta0 += error;
            errorTheta1 += error * X[i];
        }

        // Update the parameters
        theta0 -= alpha * (1.0 / m) * errorTheta0;
        theta1 -= alpha * (1.0 / m) * errorTheta1;

        //std::cout<< errorTheta0 << " " << errorTheta1 << std::endl;

        //print the cost at 100 iteration each to observe the progress
        // if(iter%100==0){
        //     std::cout << "Iteration " << iter << ": Cost = " << computeCost(X,Y, theta0, theta1) << std::endl;
        // }
    }

    // Output final values of theta0 and theta1
    std::cout << "Using Vanilla Gradient Descent without multi threading:" << std::endl;
    std::cout << "Optimized Theta0: " << theta0 << std::endl;
    std::cout << "Optimized Theta1: " << theta1 << std::endl;
}

// Structure to store gradients for each thread
struct GradientResult {
    double theta0_gradient = 0.0;
    double theta1_gradient = 0.0;
};

// Function to compute gradients for a subset of the data
void computeGradient(const std::vector<double>& X, const std::vector<double>& Y, 
                     double theta0, double theta1, int start, int end, GradientResult& gradResult) {
    double theta0_grad = 0.0;
    double theta1_grad = 0.0;
    int m = end - start;

    // Compute gradients for the subset
    for (int i = start; i < end; ++i) {
        double prediction = theta0 + theta1 * X[i];
        double error = prediction - Y[i];
        theta0_grad += error;
        theta1_grad += error * X[i];
    }

    // Accumulate gradients in the thread's local storage
    gradResult.theta0_gradient = theta0_grad / m;
    gradResult.theta1_gradient = theta1_grad / m;
}

// Function to perform vanilla gradient descent with multi-threading
void vanilla_gradient_descent_with_multithreading(std::vector<double>& X, std::vector<double>& Y) {
    double theta0 = 0.0;
    double theta1 = 0.0;
    double alpha = 0.0001;  // Learning rate
    int iterations = 100000;
    int m = X.size();
    int num_threads = 4;  // Number of threads
    int chunk_size = m / num_threads;

    for (int iter = 0; iter < iterations; ++iter) {
        // Each thread will store its gradients here
        std::vector<GradientResult> thread_gradients(num_threads);
        std::vector<std::thread> threads;

        // Split the data across threads and start them
        for (int i = 0; i < num_threads; ++i) {
            int start = i * chunk_size;
            int end = (i == num_threads - 1) ? m : start + chunk_size;
            threads.push_back(std::thread(computeGradient, std::cref(X), std::cref(Y), theta0, theta1, start, end, std::ref(thread_gradients[i])));
        }

        // Wait for all threads to finish
        for (auto& th : threads) {
            if (th.joinable()) {
                th.join();
            }
        }

        // Aggregate the gradients computed by each thread
        double sum_grad0 = 0.0;
        double sum_grad1 = 0.0;
        for (const auto& grad : thread_gradients) {
            sum_grad0 += grad.theta0_gradient;
            sum_grad1 += grad.theta1_gradient;
        }

        // Update theta0 and theta1
        theta0 -= alpha * sum_grad0;
        theta1 -= alpha * sum_grad1;
    }

    std::cout << "Using Vanilla Gradient Descent with multithreading" << std::endl;
    std::cout << "Optimized Theta0: " << theta0 << std::endl;
    std::cout << "Optimized Theta1: " << theta1 << std::endl;
}
