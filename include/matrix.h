#pragma once
#include <vector>

class Matrix {
    private:

        std::vector<double> data;
        int cols;
        int rows;

    public:

        Matrix();
        Matrix(int rows, int cols);
        Matrix(int rows, int cols, std::vector<double> data);

        double& operator()(int row, int col);
        double operator()(int row, int col) const;

        int getRows() const;
        int getCols() const;

        std::vector<double> normalize();
};
