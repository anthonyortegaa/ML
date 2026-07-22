#pragma once
#include <vector>
#include "matrix.h"

namespace logreg {

std::vector<double> predict(Matrix mtrx, std::vector<double> weight, double bias);
double compute_loss(std::vector<double> predictions, std::vector<double> y);
std::pair<std::vector<double>, double> compute_gradients(Matrix mtrx, std::vector<double> y,
                                                        std::vector<double> predictions);
double sigmoid(double z);

}
