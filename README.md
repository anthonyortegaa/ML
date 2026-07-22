# ML from Scratch (C++)

| Model | Task | Data |
|---|---|---|
| Linear regression | Predict house prices (continuous) | `data/housing.csv` — 50 fake houses |
| Logistic regression | Predict loan default (binary) | `data/loans.csv` — 200 fake applicants |
| Softmax regression | Predict wine cultivar (3 classes) | `data/wine.csv` — UCI Wine, 178 real wines |

## Build and Run

```
cmake -B build
cmake --build build
```

Run from the repo root so the CSV paths resolve. The Visual Studio generator (the default on Windows) puts the binaries in a `Debug/` subfolder:

```
./build/Debug/linear_main.exe
./build/Debug/logistic_main.exe
./build/Debug/softmax_main.exe
```

Single-config generators like Ninja or MinGW Makefiles place them directly in `build/` instead (e.g. `./build/linear_main.exe`).

## The Models

### Linear regression — house prices

Predicts a continuous value as a weighted sum of features:

```
prediction = X·w + b
```

Loss is mean squared error. Gradient descent steps every weight against its gradient each epoch; bigger errors produce bigger corrections. Features: size, beds, baths, age, distance to city, garage, school rating.

### Logistic regression — loan default

Same linear combination, wrapped in a sigmoid to produce a probability:

```
sigmoid(z) = 1 / (1 + e^(-z)),  z = X·w + b
```

Loss is binary cross-entropy (`-log(p)` when y=1, `-log(1-p)` when y=0), which punishes confident wrong answers heavily. A 0.5 threshold turns the probability into DEFAULT/REPAY. Features: credit score, annual income, debt-to-income, years employed.

### Softmax regression — wine cultivar

Generalizes logistic regression to K classes: K weight vectors and K biases produce K scores, and softmax turns them into a probability distribution:

```
softmax(z)_k = e^(z_k) / sum of e^(z_j) over all j
```

Labels are one-hot so the gradient is a clean `prediction - label` subtraction. Loss is cross-entropy of the true class. The UCI Wine data has 13 chemical measurements per wine (alcohol, malic acid, flavanoids, proline, ...) and three cultivars balanced 59/71/48.

## Shared pieces

- **Matrix** (`include/matrix.h`, `matrix.cpp`) — flat `std::vector<double>` storage with `operator()(row, col)` indexing. `normalize()` scales each column by its largest absolute value into [-1, 1] and returns the per-column scale factors, which the mains reuse to normalize user input at prediction time.
- **Metrics** (`include/metrics.h`, `metrics.cpp`) — `argMax`, plus two `accuracy_check` overloads: binary (threshold at 0.5) and multi-class (argmax of prediction vs argmax of one-hot label).
- **CSV loader** (`include/csv_loader.h`, `csv_loader.cpp`) — one `load_csv` for all three datasets, with flags for a header row and for whether the label is the first or last column, plus a one-hot overload for multi-class labels.
