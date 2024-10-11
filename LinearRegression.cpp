#include <iostream>
#include <vector>
#include <chrono>
#include "Dataset.h"
#include "VGD.h"
#include "SGD.h"
#include "AGD.h"

int main()
{
    std::vector<double>X;
    std::vector<double>Y;
    
    createData(X,Y); // from Dataset

    auto start_vgd_nomt = std::chrono::high_resolution_clock::now();
    vanilla_gradient_descent_without_multithreading(X,Y);
    auto end_vgd_nomt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_single = end_vgd_nomt - start_vgd_nomt;
    std::cout << "Time taken for single-threaded version Vanilla Gradient Descent: " << duration_single.count() << " seconds." << std::endl;

    auto start_vgd_mt = std::chrono::high_resolution_clock::now();
    vanilla_gradient_descent_with_multithreading(X,Y);
    auto end_vgd_mt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_vgd_mt = end_vgd_mt - start_vgd_mt;
    std::cout<< "Time taken for multi threaded Vanilla Gradient Descent : " << duration_vgd_mt.count() << std::endl;

    auto start_sgd_nomt = std::chrono::high_resolution_clock::now();
    stochastic_gradient_descent_without_multithreading(X,Y);
    auto end_sgd_nomt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_sgd_nomt = end_sgd_nomt - start_sgd_nomt;
    std::cout << "Time taken for single-threaded version Stochastic Gradient Descent : " << duration_sgd_nomt.count() << " seconds." << std::endl;

    auto start_sgd_mt = std::chrono::high_resolution_clock::now();
    stochastic_gradient_descent_with_multithreading(X,Y);
    auto end_sgd_mt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_sgd_mt = end_sgd_mt - start_sgd_mt;
    std::cout<< "Time taken for multi threaded Stochastic Gradient Descent : " << duration_sgd_mt.count() << std::endl;

    auto start_agd = std::chrono::high_resolution_clock::now();
    accelerated_gradient_descent(X,Y);
    auto end_agd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_agd = end_agd - start_agd;
    std::cout<< "Time taken for Accelerated Gradient Descent : " << duration_agd.count() << std::endl;
}