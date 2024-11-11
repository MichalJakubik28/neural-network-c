#include "../matrix/matrix.h"

double relu(double input);
double relu_prime(double input);
void softmax(double *vec, double *out, int size);
void he_init(Matrix *layer, int n_in);
void glorot_init(Matrix *layer, int n_in, int n_out);