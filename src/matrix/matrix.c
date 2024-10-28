#include <stdio.h>
#include "matrix.h"
#include <stdlib.h>
#include <string.h>

void print_matrix(double *matrix, int img_size) {
    for (int i = 0; i < img_size*img_size; i++) {
        if (i % (img_size) == 0) {
            putc('\n', stdout);
        }
        printf("%s%d ", matrix[i] >= 100 ? "" : matrix[i] >= 10 ? " " : "  ", (int) matrix[i]);
    }
}

Matrix* matrix_create(int rows, int cols) {
    Matrix *matrix = malloc(sizeof(Matrix));
    if (matrix == NULL) {
        return NULL;
    }
    double **data = malloc(rows * sizeof(double*));
    if (data == NULL) {
        free(matrix);
        return NULL;
    }
    for (int i = 0; i < rows; i++) {
        double *col = malloc(cols * sizeof(double));
        if (col == NULL) {
            for (int j = 0; j < i; j++) {
                free(data[j]);
                free(data);
                free(matrix);
                return NULL;
            }
        }
        data[i] = col;
    }
    matrix->cols = cols;
    matrix->rows = rows;
    matrix->data = data;
    return matrix;
}

void matrix_free(Matrix *matrix) {
    for (int i = 0; i < matrix->rows; i++) {
        free(matrix->data[i]);
    }
    free(matrix->data);
    free(matrix);
}

/*Apply function to each element in a matrix.*/
void matrix_apply(Matrix *matrix, double (*func)(double)) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            matrix->data[i][j] = func(matrix->data[i][j]);
        }
    }
}

void matrix_multiply(Matrix *a, Matrix *b) {
    return;
}

/*Dot product of <vec> and individual rows of <matrix>.*/
void matrix_dot(double *vec, Matrix *matrix, double *out) {
    memset(out, 0, matrix->rows);
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            out[i] += matrix->data[i][j] * vec[j];
        }
    }
}