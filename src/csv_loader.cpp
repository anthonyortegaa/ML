#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include "csv_loader.h"
#include "matrix.h"

// reads every row of csv doubles, pulls out the label column,
// and flattens the rest into a Matrix
static void load_rows(const std::string& filename, Matrix& mtrx, std::vector<double>& labels,
                    bool has_header, bool label_first) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "Could not open %s (are you running from the repo root?)\n", filename.c_str());
        std::exit(1);
    }
    std::string line;
    std::vector<std::vector<double>> data;

    if (has_header) {
        std::getline(file, line);
    }

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::vector<double> row;
        std::stringstream ss(line);
        std::string val;

        while (std::getline(ss, val, ',')) {
            row.push_back(std::stod(val));
        }

        if (label_first) {
            labels.push_back(row.front());
            row.erase(row.begin());
        } else {
            labels.push_back(row.back());
            row.pop_back();
        }
        data.push_back(row);
    }

    // flatten 2d vect to 1d
    std::vector<double> flat_data;
    for (const auto& row : data) {
        for (double val : row) {
            flat_data.push_back(val);
        }
    }
    mtrx = Matrix(data.size(), data[0].size(), flat_data);
    file.close();
}

void load_csv(const std::string& filename, Matrix& mtrx, std::vector<double>& y,
            bool has_header, bool label_first) {
    load_rows(filename, mtrx, y, has_header, label_first);
}

void load_csv(const std::string& filename, Matrix& mtrx, std::vector<std::vector<double>>& y,
            int num_classes, bool has_header, bool label_first) {
    std::vector<double> labels;
    load_rows(filename, mtrx, labels, has_header, label_first);

    for (double label : labels) {
        std::vector<double> one_hot(num_classes, 0.0);
        one_hot[static_cast<int>(label) - 1] = 1.0;
        y.push_back(one_hot);
    }
}
