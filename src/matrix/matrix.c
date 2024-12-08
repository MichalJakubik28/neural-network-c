#include "matrix.h"

/*Print matrix to stdout.*/
void matrix_print(Matrix *matrix) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            printf("%.8f ", matrix->data[i][j]);
        }
        printf("\n");
    }
}

/*Create a matrix with given number of rows and cols.*/
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

/*Free the given matrix.*/
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

/*Add two matrices together and store result in <out>*/
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

/*Multiply elements in matrix by constant.*/
void matrix_multiply_by_constant(Matrix *matrix, double constant) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            matrix->data[i][j] *= constant;
        }
    }
}

/*Set values in <matrix> to <value>.*/
void matrix_set(Matrix *matrix, double value) {
    for (int i = 0; i < matrix->rows; i++) {
        memset(matrix->data[i], value, matrix->cols * sizeof(double));
    }
}

/*Transpose <matrix> and store result in <out>.*/
void matrix_transpose(Matrix *matrix, Matrix *out) {
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            out->data[j][i] = matrix->data[i][j];
        }
    }
}

/*Multiply two matrices, store result in <out>*/
void matrix_multiply(Matrix *a, Matrix *b, Matrix *out) {
    #pragma omp parallel for collapse(2) shared(out) num_threads(16)
    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            out->data[i][j] = 0;
            for (int k = 0; k < a->cols; k++) {
                out->data[i][j] += a->data[i][k] * b->data[k][j];
            }
        }
    }
}

/*Apply function to all rows of a matrix, store result in <out>.*/
void matrix_apply_row_wise(Matrix *matrix, Matrix *out, void (*func)(double*, double*, int)) {
    for (int i = 0; i < matrix->rows; i++) {
        func(matrix->data[i], out->data[i], matrix->cols);
    }
}

/*Copy <src> matrix to <dest>.*/
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