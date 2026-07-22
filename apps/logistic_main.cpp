#include <iostream>
#include <vector>
#include <cstdio>
#include "logistic_regression.h"
#include "metrics.h"
#include "matrix.h"
#include "csv_loader.h"

int main() {
    Matrix mtrx;
    std::vector<double> y;
    load_csv("data/loans.csv", mtrx, y);

    std::vector<double> mtrx_maxes = mtrx.normalize();

    std::vector<double> weights(mtrx.getCols(), 0.0);
    double bias = 0.0;
    double learning_rate = 0.1;
    int epoch = 10000;

    for (int i{}; i <= epoch; ++i) {
        std::vector<double> predictions = logreg::predict(mtrx, weights, bias);
        double loss = logreg::compute_loss(predictions, y);
        std::pair<std::vector<double>, double> gradients = logreg::compute_gradients(mtrx, y, predictions);
        for (int j{}; j < weights.size(); ++j) {
            weights[j] = weights[j] - learning_rate * gradients.first[j];
        }
        bias = bias - learning_rate * gradients.second;
        double accuracy = accuracy_check(predictions, y);

        if (i % 1000 == 0) {
            printf("Stats at epoch: %d\n", i);
            printf("Loss: %.3f\n", loss);
            printf("Accuracy: %.3f\n", accuracy);
            for (int j{}; j < weights.size(); ++j) {
                printf("weight[%d]: %.3f\n", j, weights[j]);
            }
            printf("Bias: %.3f\n", bias);
        }
    }

    // Interactive predictions
    std::vector<std::string> feature_names = {"credit_score", "annual_income(k)",
                                            "debt_to_income(dec)", "years_employed"};

    while (true) {
        printf("\nEnter loan applicant info (or -1 to quit):\n");

        std::vector<double> input(mtrx.getCols());
        bool quit = false;

        for (int j = 0; j < mtrx.getCols(); ++j) {
            printf("%s: ", feature_names[j].c_str());
            std::cin >> input[j];
            if (input[j] == -1) {
                quit = true;
                break;
            }
        }
        if (quit) break;

        for (int j = 0; j < mtrx.getCols(); ++j) {
            input[j] = input[j] / mtrx_maxes[j];
        }

        double z = bias;
        for (int j = 0; j < mtrx.getCols(); ++j) {
            z = z + (input[j] * weights[j]);
        }
        double probability = logreg::sigmoid(z);

        // Classify with 0.5 threshold
        if (probability >= 0.5) {
            printf("Prediction: DEFAULT (probability: %.1f%%)\n", probability * 100);
        } else {
            printf("Prediction: REPAY (default probability: %.1f%%)\n", probability * 100);
        }
    }

    return 0;
}