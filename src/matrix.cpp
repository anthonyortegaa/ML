#include <vector>
#include <cmath>
#include "matrix.h"

Matrix::Matrix() {
    this->rows = 0;
    this->cols = 0;
}

Matrix::Matrix(int rows, int cols) {
    this->rows = rows;
    this->cols = cols;
    this->data = std::vector<double> (rows * cols, 0.0);
}

Matrix::Matrix(int rows, int cols, std::vector<double> data) {
    this->rows = rows;
    this->cols = cols;
    this->data = data;
}

double& Matrix::operator()(int row, int col) {
    return data[row * cols + col];
}

double Matrix::operator()(int row, int col) const {
    return data[row * cols + col];
}

int Matrix::getRows() const {
    return this->rows;
}

int Matrix::getCols() const {
    return this->cols;
}

// scales each column by its largest absolute value so entries land in [-1, 1]
// (dividing by a signed max would flip the sign of all-negative columns)
std::vector<double> Matrix::normalize() {
    std::vector<double> maxColVals;
    for (int col{}; col < cols; ++col) {
        double max_val = 0.0;
        for (int row{}; row < rows; ++row) {
            if(std::abs((*this)(row,col)) > max_val) {
                max_val = std::abs((*this)(row,col));
            }
        }
        if (max_val != 0.0) {
            for (int row{}; row < rows; ++row) {
                (*this)(row,col) = (*this)(row,col) / max_val;
            }
        }

        maxColVals.push_back(max_val);
    }

    return maxColVals;
}
