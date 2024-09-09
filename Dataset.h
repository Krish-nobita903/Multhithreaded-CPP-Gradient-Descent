#ifndef DATASET_H
#define DATASET_H

#include <vector>

void generateData(std::vector<double>& X, std::vector<double>& Y, int dataSize, double trueTheta0, double trueTheta1);
void createData(std::vector<double>& X, std::vector<double>& Y);

#endif // DATASET_H
