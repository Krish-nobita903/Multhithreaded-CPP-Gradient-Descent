#ifndef HELPER_H
#define HELPER_H

#include <vector>

double hypothesis(double theta0, double theta1, double x);
double computeCost(std::vector<double>&X,std::vector<double>&Y, double theta0, double theta1);

#endif