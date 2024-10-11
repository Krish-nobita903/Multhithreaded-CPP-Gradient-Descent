#include<iostream>
#include<vector>
#include<cmath>
#include<ctime>
#include<cstdlib>
#include "Dataset.h"

void generateData(std::vector<double>&X,std::vector<double>&Y,int dataSize,double trueTheta0,double trueTheta1)
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    for(int i=0;i<dataSize;i++)
    {
        double x = static_cast<double>(i); // X values
        double noise = (std::rand()%100)/100.0 - 0.5; // random noise between -0.5 to 0.5
        double y = trueTheta0 + trueTheta1*x + noise; // Yvalues
        X.push_back(x);
        Y.push_back(y);
    }
}

void createData(std::vector<double>&X,std::vector<double>&Y)
{
    // Assuming a simple linear regression y = a+bx
    int dataSize = 100;
    double trueTheta0 = 5.5;  // True intercept (now a double)
    double trueTheta1 = 2.3;  // True slope (now a double)

    generateData(X, Y, dataSize, trueTheta0, trueTheta1);
}