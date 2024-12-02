#!/bin/bash
## change this file to your needs

echo "Adding some modules"

# module add gcc-10.2

echo "#################"
echo "    COMPILING    "
echo "#################"

gcc -fopenmp -Ofast src/main.c -o src/main src/utils/csv_to_img.c src/matrix/matrix.c src/matrix/vec_ops.c src/weight_ops/weight_ops.c -lm

## dont forget to use comiler optimizations (e.g. -O3 or -Ofast)
# g++ -Wall -std=c++17 -O3 src/main.cpp src/file2.cpp -o network


echo "#################"
echo "     RUNNING     "
echo "#################"

src/main

echo "train accuracy:"
python3 evaluator/evaluate.py train_predictions.csv data/fashion_mnist_train_labels.csv

echo "test accuracy:"
python3 evaluator/evaluate.py test_predictions.csv data/fashion_mnist_test_labels.csv

## use nice to decrease priority in order to comply with aisa rules
## https://www.fi.muni.cz/tech/unix/computation.html.en
## especially if you are using multiple cores
# nice -n 19 ./network
