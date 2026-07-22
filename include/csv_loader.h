#pragma once
#include <vector>
#include <string>
#include "matrix.h"

// loads features into mtrx and the label column into y
void load_csv(const std::string& filename, Matrix& mtrx, std::vector<double>& y,
            bool has_header = true, bool label_first = false);

// multi-class variant
void load_csv(const std::string& filename, Matrix& mtrx, std::vector<std::vector<double>>& y,
            int num_classes, bool has_header = false, bool label_first = true);
