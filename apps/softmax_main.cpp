#include <iostream>
#include <vector>
#include <cstdio>
#include <string>
#include "softmax_regression.h"
#include "metrics.h"
#include "matrix.h"
#include "csv_loader.h"

const int NUM_CLASSES = 3;

int main() {
    Matrix mtrx;
    std::vector<std::vector<double>> y;
    // UCI Wine: no header row, label is the FIRST column, labels are 1/2/3
    load_csv("data/wine.csv", mtrx, y, NUM_CLASSES);

    printf("Loaded %d rows, %d features\n\n", mtrx.getRows(), mtrx.getCols());

    std::vector<double> mtrx_maxes = mtrx.normalize();

    std::vector<std::vector<double>> weights(NUM_CLASSES,
                                            std::vector<double>(mtrx.getCols(), 0.0));
    std::vector<double> bias(NUM_CLASSES, 0.0);

    double learning_rate = 0.1;
    int epoch = 10000;

    for (int i = 0; i <= epoch; ++i) {
        std::vector<std::vector<double>> predictions = softreg::predict(mtrx, weights, bias);
        double loss = softreg::compute_loss(predictions, y);
        std::pair<std::vector<std::vector<double>>, std::vector<double>> gradients =
                                                softreg::compute_gradients(mtrx, y, predictions);
        for (int v{}; v < gradients.second.size(); ++v) {
            bias[v] -= learning_rate * gradients.second[v];
        }

        for (int v{}; v < gradients.first.size(); ++v) {
            for (int j{}; j <  mtrx.getCols(); ++j) {
                weights[v][j] -= learning_rate * gradients.first[v][j];
            }
        }


        double accuracy = accuracy_check(predictions, y);

        if (i % 1000 == 0) {
            printf("Stats at epoch: %d\n", i);
            printf("Loss: %.4f\n", loss);
            printf("Accuracy: %.3f\n", accuracy);
            for (int k = 0; k < NUM_CLASSES; ++k) {
                printf("  cultivar %d bias: %.3f | weights:", k + 1, bias[k]);
                for (int j = 0; j < mtrx.getCols(); ++j) {
                    printf(" %.3f", weights[k][j]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    // Interactive predictions
    std::vector<std::string> feature_names = {
        "alcohol", "malic_acid", "ash", "alkalinity_of_ash", "magnesium",
        "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
        "color_intensity", "hue", "od280_od315", "proline"};
    std::vector<std::string> class_names = {"CULTIVAR 1", "CULTIVAR 2", "CULTIVAR 3"};

    while (true) {
        printf("\nEnter wine measurements (or -1 to quit):\n");

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

        std::vector<double> scores;
        for (int k = 0; k < NUM_CLASSES; ++k) {
            double score = bias[k];
            for (int j = 0; j < mtrx.getCols(); ++j) {
                score = score + weights[k][j] * input[j];
            }
            scores.push_back(score);
        }

        std::vector<double> probabilities = softreg::softmax(scores);
        int predicted = argMax(probabilities);

        printf("Prediction: %s\n", class_names[predicted].c_str());
        for (int k = 0; k < NUM_CLASSES; ++k) {
            printf("  %s: %.1f%%\n", class_names[k].c_str(), probabilities[k] * 100);
        }
    }

    return 0;
}