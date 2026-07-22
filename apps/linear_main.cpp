#include <iostream>
#include <vector>
#include <cstdio>
#include <cmath>
#include "linear_regression.h"
#include "matrix.h"
#include "csv_loader.h"


double normalize_vector(std::vector<double>& vect) {
    double max = 0.0;
    for (const double val : vect) {
        max = (val > max) ? val : max;
    }

    for (double& val : vect) {
        val = val / max;
    }

    return max;
}

int main() {
    // house features with price (in thousands) as the label
    Matrix X;
    std::vector<double> y;
    load_csv("data/housing.csv", X, y);

    std::vector<double> X_maxes = X.normalize();
    double y_max = normalize_vector(y);

    std::vector<double> weight(X.getCols(), 0.0);
    double bias = 0.0;
    double learning_rate = .1;
    int epochs = 5000;

    // gradiaent descent
    for (int i{}; i <= epochs; ++i) {
        std::vector<double> predictions = linreg::predict(X, weight, bias);
        double loss = linreg::compute_loss(predictions, y);
        std::pair<std::vector<double>, double> grads = linreg::compute_gradients(X, y, predictions);
        for (int j{}; j < weight.size(); ++j) {
            weight[j] = weight[j] - learning_rate * grads.first[j];
        }
        bias = bias - learning_rate * grads.second;

        if (i % 100 == 0) {
            printf("Stats at epoch: %d\n", i);
            printf("Loss: %.3f\n", loss);
            printf("RMSE: $%.1fk\n", sqrt(loss) * y_max);
            for (int j{}; j < weight.size(); ++j) {
                printf("Weight[%d]: %.3f\n", j, weight[j]);
            }
            printf("Bias: %.3f\n\n", bias);
        }
    }

    std::vector<double> final_predictions = linreg::predict(X, weight, bias);
    double final_loss = linreg::compute_loss(final_predictions, y);
    printf("Final RMSE: $%.1fk (typical prediction error on training data)\n", sqrt(final_loss) * y_max);

    //user can have a prediction made for them in terminal
    char choice;
    printf("Would you like to make a prediction? (y/n): ");
    scanf(" %c", &choice);

    while (choice == 'y' || choice == 'Y') {
        Matrix new_house(1, 7, {0,0,0,0,0,0,0});
        double size, beds, baths, age, dist, garage, school;

        printf("Enter size (sqft): ");
        scanf("%lf", &size);
        printf("Enter number of beds: ");
        scanf("%lf", &beds);
        printf("Enter number of baths: ");
        scanf("%lf", &baths);
        printf("Enter age of house (yrs): ");
        scanf("%lf", &age);
        printf("Enter distance to city (mi): ");
        scanf("%lf", &dist);
        printf("Enter garage (0 = no, 1 = yes): ");
        scanf("%lf", &garage);
        printf("Enter school rating (1-10): ");
        scanf("%lf", &school);

        new_house(0,0) = size;
        new_house(0,1) = beds;
        new_house(0,2) = baths;
        new_house(0,3) = age;
        new_house(0,4) = dist;
        new_house(0,5) = garage;
        new_house(0,6) = school;

        for (int col{}; col < 7; ++col) {
            if (X_maxes[col] != 0.0) {
                new_house(0, col) = new_house(0, col) / X_maxes[col];
            }
        }

        std::vector<double> result = linreg::predict(new_house, weight, bias);
        printf("Predicted price: $%.0fk\n", result[0] * y_max);

        printf("\nWould you like to make another prediction? (y/n): ");
        scanf(" %c", &choice);
    }

    return 0;
}
