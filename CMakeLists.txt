cmake_minimum_required(VERSION 3.16)
project(ml CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_library(mlcore STATIC
    src/matrix.cpp
    src/metrics.cpp
    src/csv_loader.cpp
)

add_executable(linear_main apps/linear_main.cpp src/models/linear_regression.cpp)
add_executable(logistic_main apps/logistic_main.cpp src/models/logistic_regression.cpp)
add_executable(softmax_main apps/softmax_main.cpp src/models/softmax_regression.cpp)

target_link_libraries(linear_main mlcore)
target_link_libraries(logistic_main mlcore)
target_link_libraries(softmax_main mlcore)
