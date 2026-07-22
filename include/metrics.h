#pragma once
#include <vector>

// index of the largest element
int argMax(std::vector<double> scores);

// binary classification
double accuracy_check(std::vector<double> predictions, std::vector<double> y);

// multi-class
double accuracy_check(std::vector<std::vector<double>> predictions, std::vector<std::vector<double>> y);
