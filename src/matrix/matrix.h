#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int rows;
    int cols;
    double **data;
} Matrix;

void print_matrix(double *matrix, int img_size);
void matrix_print(Matrix *matrix);
Matrix *matrix_create(int rows, int cols);
void matrix_free(Matrix *matrix);
void matrix_apply(Matrix *matrix, double (*func)(double), Matrix *out);
void matrix_dot(double *vec, Matrix *matrix, double *out);
void matrix_add(Matrix *a, Matrix *b, Matrix *out);
void matrix_multiply_by_constant(Matrix *matrix, double constant);
void matrix_set(Matrix *matrix, double value);
void matrix_transpose(Matrix *matrix, Matrix *out);
void matrix_multiply(Matrix *a, Matrix *b, Matrix *out);
void matrix_apply_row_wise(Matrix *matrix, Matrix *out, void (*func)(double*, double*, int));
void matrix_copy(Matrix *src, Matrix *dest);