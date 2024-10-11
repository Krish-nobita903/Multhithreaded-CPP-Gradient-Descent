#ifndef SGD_H
#define SGD_H

#include <vector>

void stochastic_gradient_descent_without_multithreading(std::vector<double>& X, std::vector<double>& Y);
void stochastic_gradient_descent_with_multithreading(std::vector<double>& X, std::vector<double>& Y);

#endif