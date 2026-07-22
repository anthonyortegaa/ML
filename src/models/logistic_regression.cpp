#include <vector>
#include <cmath>
#include "logistic_regression.h"
#include "matrix.h"

namespace logreg {

double sigmoid(double z) {
    return 1 / (1 + exp(-z));
}
std::vector<double> predict(Matrix mtrx, std::vector<double> weights, double bias) {
    std::vector<double> predictions;
    for (int row{}; row < mtrx.getRows(); ++row) {
        double sum = 0.0;
        for (int col{}; col < mtrx.getCols(); ++col) {
            sum = sum + (mtrx(row, col) * weights[col]);
        }
        sum = sum + bias;
        predictions.push_back(sigmoid(sum));
    }

    return predictions;
}


double compute_loss(std::vector<double> predictions, std::vector<double> y) {
    double loss = 0.0;
    int n = predictions.size();
    for (int i{}; i < n; ++i) {
        loss = (y[i]) ? loss - log(predictions[i]): loss - log(1 - predictions[i]);
    }

    return loss / n;
}

std::pair<std::vector<double>, double> compute_gradients(Matrix mtrx, std::vector<double> y,
                                                        std::vector<double> predictions) {
    std::vector<double> pred_w(mtrx.getCols(), 0.0);
    double pred_b = 0.0;
    for (int row{}; row < mtrx.getRows(); ++row) {
        double residual = predictions[row] - y[row];
        for (int col{}; col < mtrx.getCols(); ++col) {
            pred_w[col] = pred_w[col] + (mtrx(row,col) * residual);
        }
        pred_b = pred_b + residual;
    }

    int n = predictions.size();
    for (int i{}; i < pred_w.size(); ++i) {
        pred_w[i] = pred_w[i] / n;
    }
    pred_b = pred_b / n;

    return {pred_w, pred_b};
}

}
