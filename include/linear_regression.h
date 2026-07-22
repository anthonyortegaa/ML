#pragma once
#include <vector>
#include "matrix.h"

namespace linreg {

std::vector<double> predict(Matrix mtrx, std::vector<double> w, double b);
double compute_loss(std::vector<double> predictions, std::vector<double> y);
std::pair<std::vector<double>, double> compute_gradients(Matrix mtrx, std::vector<double> y,
                                            std::vector<double> predictions);

}
