#pragma once
#include <vector>
#include "matrix.h"

namespace softreg {

std::vector<std::vector<double>> predict(Matrix mtrx, std::vector<std::vector<double>> weights,
                                        std::vector<double> bias);
double compute_loss(std::vector<std::vector<double>> predictions, std::vector<std::vector<double>> y);
std::pair<std::vector<std::vector<double>>, std::vector<double>> compute_gradients(Matrix mtrx,
                                                            std::vector<std::vector<double>> y,
                                                            std::vector<std::vector<double>> predictions);
std::vector<double> softmax(const std::vector<double>& scores);

}
