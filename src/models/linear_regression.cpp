#include <vector>
#include <utility>
#include <cmath>
#include "linear_regression.h"
#include "matrix.h"

namespace linreg {

std::vector<double> predict(Matrix mtrx, std::vector<double> w, double b) {

    std::vector<double> predictions;
    for (int row{}; row < mtrx.getRows(); ++row) {
        double sum = 0;
        for (int col{}; col < mtrx.getCols(); ++col) {
            sum = sum + (mtrx(row,col) * w[col]);
        }
        predictions.push_back(sum + b);
    }

    return predictions;
}

double compute_loss(std::vector<double> predictions, std::vector<double> y) {
    
    double loss = 0.0;
    int n = predictions.size();
    for (int i{}; i < n; ++i) {
        double residual = predictions[i] - y[i];
        loss = loss + (residual * residual);
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
            pred_w[col] = pred_w[col] + (residual * mtrx(row,col));
        }
        pred_b = pred_b + residual;
    }

    int n = predictions.size();
    for (int i{}; i < pred_w.size(); ++i) {
        pred_w[i] = pred_w[i] /n;
    }
    pred_b = pred_b / n;

    return {pred_w, pred_b};
}

}

