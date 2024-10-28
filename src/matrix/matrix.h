typedef struct {
    int rows;
    int cols;
    double **data;
} Matrix;

void print_matrix(double *matrix, int img_size);
Matrix *matrix_create(int rows, int cols);
void matrix_free(Matrix *matrix);
void matrix_apply(Matrix *matrix, double (*func)(double));
void matrix_multiply(Matrix *a, Matrix *b);
void matrix_dot(double *vec, Matrix *matrix, double *out);