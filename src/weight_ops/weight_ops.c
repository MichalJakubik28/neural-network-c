#include "weight_ops.h"

double relu(double input) {
    return input <= 0 ? 0 : input;
}

double relu_prime(double input) {
    return input <= 0 ? 0 : 1;
}

void softmax(double *vec, double *out, int size) {
    double softmax_sum = 0;
    for (int i = 0; i < size; i++) {
        softmax_sum += exp(vec[i]);
    }
    for (int i = 0; i < size; i++) {
        out[i] = exp(vec[i])/softmax_sum;
    }
}

void he_init(Matrix *layer, int n_in) {
    double stddev = sqrt(2.0 / n_in);
    for (int i = 0; i < layer->rows; i++) {
        for (int j = 0; j < layer->cols; j++)
            layer->data[i][j] = stddev * ((double)rand() / RAND_MAX * 2 - 1); // Uniform around 0
            // layer->data[i][j] = 0.001;
    }
}

void glorot_init(Matrix *layer, int n_in, int n_out) {
    double limit = sqrt(6.0 / (n_in + n_out));
    for (int i = 0; i < layer->rows; i++) {
        for (int j = 0; j < layer->cols; j++) {
            layer->data[i][j] = ((double)rand() / RAND_MAX) * 2 * limit - limit; // Uniform in [-limit, limit]
            // layer->data[i][j] = 0.001;
        }
    }
}