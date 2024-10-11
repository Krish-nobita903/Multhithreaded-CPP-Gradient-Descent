#ifndef VGD_H
#define VGD_H

#include <vector>

void vanilla_gradient_descent_without_multithreading(std::vector<double>&X,std::vector<double>&Y);
void vanilla_gradient_descent_with_multithreading(std::vector<double>&X,std::vector<double>&Y);

#endif // DATASET_H