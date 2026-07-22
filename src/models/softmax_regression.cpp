#include <vector>
#include <cmath>
#include <limits>
#include <utility>
#include "softmax_regression.h"
#include "matrix.h"
#include "metrics.h"

namespace softreg {

//softmax_helper
std::vector<double> softmax(const std::vector<double>& scores) {
    double max_val = std::numeric_limits<double>::lowest();
    for (double val : scores) {
        max_val = (max_val < val)? val : max_val;
    }

    double sum = 0.0;
    std::vector<double> res;
    for (double val : scores) {
        double e = std::exp(val - max_val);
        res.push_back(e);
        sum = sum + e;
    }

    for (double& val : res) {
        val = val / sum;
    }

    return res;
}
//predict
std::vector<std::vector<double>> predict(Matrix mtrx,
                                        std::vector<std::vector<double>> weights,
                                        std::vector<double> bias) {
    std::vector<std::vector<double>> predictions;
    for (int row{}; row < mtrx.getRows(); ++row) {
        std::vector<double> scores;
        for (int k{}; k < weights.size(); ++k) {
            double score = bias[k];
            for (int col{}; col < mtrx.getCols(); ++col) {
                score = score + weights[k][col] * mtrx(row, col);
            }
            scores.push_back(score);
        }
        predictions.push_back(softmax(scores));
    }

    return predictions;
}
//compute_loss
double compute_loss(std::vector<std::vector<double>> predictions,
                            std::vector<std::vector<double>> y) {
    double loss = 0.0;
    for (int i{}; i < y.size(); ++i) {
        int true_class = argMax(y[i]);
        loss = loss - std::log(predictions[i][true_class] + 1e-15);
    }

    return loss / y.size();
}
//compute_gradients
std::pair<std::vector<std::vector<double>>, std::vector<double>> compute_gradients(Matrix mtrx,
                                                    std::vector<std::vector<double>> y,
                                                    std::vector<std::vector<double>> predictions) {

    int K = y[0].size();
    int feat = mtrx.getCols();

    std::vector<std::vector<double>> weight_grads(K, std::vector<double>(feat, 0.0));
    std::vector<double> bias_grads(K, 0.0);

    for (int row{}; row < mtrx.getRows(); ++row) {
        for (int k{}; k < K; ++k) {
            double error = predictions[row][k] - y[row][k];
            bias_grads[k] += error;
            for (int col{}; col < feat; ++col) {
                weight_grads[k][col] += error * mtrx(row,col);
            }
        }
    }

    int n = mtrx.getRows();
    for (double& val : bias_grads) {
        val = val / n;
    }

    for (std::vector<double>& vec : weight_grads) {
        for (double& val : vec) {
            val = val / n;
        }
    }

    return {weight_grads, bias_grads};
}


}
