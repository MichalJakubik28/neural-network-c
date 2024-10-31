#include "vec_ops.h"

/*Applies function to a vector.*/
void vec_apply(double *vec, double (*func)(double), double *out, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = func(vec[i]);
    }
}