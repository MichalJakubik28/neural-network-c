#!/bin/bash

echo "#################"
echo "    COMPILING    "
echo "#################"

gcc -fopenmp -O3 -Wall src/main.c -o src/main src/utils/csv_to_img.c src/matrix/matrix.c src/weight_ops/weight_ops.c src/network/network.c -lm

## dont forget to use comiler optimizations (e.g. -O3 or -Ofast)
# g++ -Wall -std=c++17 -O3 src/main.cpp src/file2.cpp -o network

echo "#################"
echo "     RUNNING     "
echo "#################"

nice -n 19 src/main

# echo "train accuracy:"
# python3 evaluator/evaluate.py train_predictions.csv data/fashion_mnist_train_labels.csv

# echo "test accuracy:"
# python3 evaluator/evaluate.py test_predictions.csv data/fashion_mnist_test_labels.csv
