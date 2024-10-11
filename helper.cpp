#include <iostream>
#include <vector>
#include <cmath>

// Hypothesis function: h(θ) = θ0 + θ1 * x
double hypothesis(double theta0, double theta1, double x) {
    return theta0 + theta1 * x;
}

// Compute the cost function (Mean Squared Error)
double computeCost(std::vector<double>&X,std::vector<double>&Y, double theta0, double theta1) {
    double cost = 0.0;
    int m = X.size();  // Number of training examples
    for (int i = 0; i < m; ++i) {
        double h = hypothesis(theta0, theta1, X[i]);
        cost += (h - Y[i])*(h-Y[i]);
    }
    return cost / (2 * m);
}