#include <vector>
#include "metrics.h"

int argMax(std::vector<double> scores) {
    double curr = scores[0];
    int max = 0;

    for (int i{}; i < scores.size(); ++i) {
        if (scores[i] > curr) {
            curr = scores[i];
            max = i;
        }
    }

    return max;
}

double accuracy_check(std::vector<double> predictions, std::vector<double> y) {
    int n = predictions.size();
    int correct = 0;
    for (int i{}; i < n; ++i) {
        double thres = (predictions[i] >= .5) ? 1 : 0;
        if (thres == y[i]) {
            ++correct;
        }
    }

    return (double)correct / n;
}

double accuracy_check(std::vector<std::vector<double>> predictions, std::vector<std::vector<double>> y) {
    int n = predictions.size();
    int correct = 0;

    for (int i{}; i < predictions.size(); ++i) {
        int pred_check = argMax(predictions[i]);
        int y_check = argMax(y[i]);

        if (pred_check == y_check) {
            ++correct;
        }
    }

    return (double)correct / n;
}
