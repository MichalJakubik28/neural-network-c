#!/bin/bash
echo "Adding gcc..."
module add gcc
if [ $? -eq 0 ]; then echo "Imported gcc."; fi;
echo "#################"
echo "    COMPILING    "
echo "#################"

gcc -fopenmp -O3 -Wall src/main.c -o src/main src/utils/csv_to_img.c src/matrix/matrix.c src/weight_ops/weight_ops.c src/network/network.c -lm
if [ $? -eq 0 ]; then echo "Compilation successful."; fi;
echo "#################"
echo "     RUNNING     "
echo "#################"

nice -n 19 src/main
