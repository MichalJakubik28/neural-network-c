#include <stdio.h>
#include "matrix.h"

void print_matrix(double *matrix, int img_size) {
    for (int i = 0; i < img_size*img_size; i++) {
        if (i % (img_size) == 0) {
            putc('\n', stdout);
        }
        printf("%s%d ", matrix[i] >= 100 ? "" : matrix[i] >= 10 ? " " : "  ", (int) matrix[i]);
    }
}