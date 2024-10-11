#include <iostream>
#include <vector>
#include <cmath>
#include "helper.h"
#include "AGD.h"

void accelerated_gradient_descent(std::vector<double>& X, std::vector<double>& Y) {
    double alpha = 0.0001;  // Learning rate
    double beta = 0.9;     // Momentum term
    int iterations = 15000;
    double theta0 = 0.0;
    double theta1 = 0.0;
    int m = X.size();
    
    // Initialize velocity terms
    double v0 = 0.0;
    double v1 = 0.0;

    for (int iter = 0; iter < iterations; ++iter) {
        double errorTheta0 = 0.0;
        double errorTheta1 = 0.0;

        // Calculate the gradient using "look-ahead" at θ - βv
        double theta0_lookahead = theta0 - beta * v0;
        double theta1_lookahead = theta1 - beta * v1;

        for (int i = 0; i < m; ++i) {
            double error = hypothesis(theta0_lookahead, theta1_lookahead, X[i]) - Y[i];
            errorTheta0 += error;
            errorTheta1 += error * X[i];
        }

        errorTheta0 /= m;
        errorTheta1 /= m;

        // Update velocities with momentum
        v0 = beta * v0 + alpha * errorTheta0;
        v1 = beta * v1 + alpha * errorTheta1;

        // Update the parameters
        theta0 -= v0;
        theta1 -= v1;

        // Optionally, print the cost every 10000 iterations
        // if (iter % 10000 == 0) {
        //     std::cout << "Iteration " << iter << ": Cost = " << computeCost(X, Y, theta0, theta1) << std::endl;
        // }
    }
    std::cout << "Cost = " << computeCost(X, Y, theta0, theta1) << std::endl;
    // Output final values of theta0 and theta1
    std::cout << "Using Accelerated Gradient Descent:" << std::endl;
    std::cout << "Optimized Theta0: " << theta0 << std::endl;
    std::cout << "Optimized Theta1: " << theta1 << std::endl;
}