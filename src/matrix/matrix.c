#include <stdio.h>
#include "../weight_ops/weight_ops.h"
#include <stdlib.h>
#include <string.h>

void print_matrix(double *matrix, int img_size) {
    for (int i = 0; i < img_size*img_size; i++) {
        if (i % (img_size) == 0) {
            putc('\n', stdout);
        }
        printf("%s%f ", matrix[i] >= 100 ? "" : matrix[i] >= 10 ? " " : "  ", (int) matrix[i]);
    }
}

void matrix_print(Matrix *matrix) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            printf("%.8f ", matrix->data[i][j]);
        }
        printf("\n");
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
        double *col = calloc(cols, sizeof(double));
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
void matrix_apply(Matrix *matrix, double (*func)(double), Matrix *out) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            out->data[i][j] = func(matrix->data[i][j]);
        }
    }
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

void matrix_add(Matrix *a, Matrix *b, Matrix *out) {
    if (a->cols != b->cols || a->rows != b->rows) {
        printf("matrices sizes do not match!\n");
        return;
    }
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            out->data[i][j] = a->data[i][j] + b->data[i][j];
        }
    }
}

void matrix_multiply_by_constant(Matrix *matrix, double constant) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            matrix->data[i][j] *= constant;
        }
    }
}

void matrix_set(Matrix *matrix, double value) {
    for (int i = 0; i < matrix->rows; i++) {
        memset(matrix->data[i], value, matrix->cols * sizeof(double));
    }
}

void matrix_transpose(Matrix *matrix, Matrix *out) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            out->data[j][i] = matrix->data[i][j];
        }
    }
}

// void matrix_multiply(Matrix *a, Matrix *b, Matrix *out) {
//     for (int i = 0; i < out->rows; i++) {
//         for (int j = 0; j < out->cols; j++) {
//             out->data[i][j] = 0;
//             for (int k = 0; k < a->cols; k++) {
//                 double tmp = a->data[i][k] * b->data[k][j];
//                 out->data[i][j] += tmp;
//             }
//         }
//     }
// }

void matrix_multiply(Matrix *a, Matrix *b, Matrix *out) {
    // Parallelize the outermost loop using OpenMP
    #pragma omp parallel for collapse(2) shared(a, b, out)
    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            out->data[i][j] = 0;
            for (int k = 0; k < a->cols; k++) {
                out->data[i][j] += a->data[i][k] * b->data[k][j];
            }
        }
    }
}

void matrix_apply_row_wise(Matrix *matrix, Matrix *out, void (*func)(double*, double*, int)) {
    for (int i = 0; i < matrix->rows; i++) {
        func(matrix->data[i], out->data[i], matrix->cols);
    }
}

void matrix_copy(Matrix *src, Matrix *dest) {
    if (src->rows != dest->rows || src->cols != dest->cols) {
        printf("Cannot copy, sizes of matrices do not match.");
        return;
    }
    for (int i = 0; i < src->rows; i++) {
        for (int j = 0; j < src->cols; j++) {
            dest->data[i][j] = src->data[i][j];
        }
    }
}