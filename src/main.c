#include <stdio.h>
#include <math.h>
#include "utils/csv_to_img.h"
// #include "matrix/matrix.h"
#include "matrix/vec_ops.h"
#include "weight_ops/weight_ops.h"

#ifndef IMG_NUM
#define IMG_NUM 0
#endif

int main() {
    srand(42);
    // int dataset_size;
    // Image **dataset = csv_to_imgs("../data/fashion_mnist_train_vectors.csv", 28, &dataset_size);
    // print_matrix(dataset[IMG_NUM]->data, 28);
    // free_dataset(dataset, dataset_size);

    // TODO prepojit vstupy s neuronkou
    double inputs_test[4][3] = {{0,0,1}, {0,1,1}, {1,0,1}, {1,1,1}};
    double **inputs = malloc(4 * sizeof(double*));
    for (int i = 0; i < 4; i++){
        double *line = malloc(3 * sizeof(double));
        inputs[i] = line;
        inputs[i][0] = inputs_test[i][0];
        inputs[i][1] = inputs_test[i][1];
        inputs[i][2] = inputs_test[i][2];
    }
    int labels[4] = {0, 1, 1, 0};

    int index = rand() % 4;
    double *input = inputs[index];
    Matrix *hidden = matrix_create(5, 3);
    he_init(hidden, 3);
    printf("Initialized weights in hidden layer:\n");
    for (int i = 0; i < hidden->rows; i++) {
        for (int j = 0; j < hidden->cols; j++) {
            printf("%.2f ", hidden->data[i][j]);
        }
        printf("\n");
    }

    Matrix *output = matrix_create(2, 6);
    glorot_init(output, 5, 2);
    printf("Initialized weights in output layer:\n");
    for (int i = 0; i < output->rows; i++) {
        for (int j = 0; j < output->cols; j++) {
            printf("%.2f ", output->data[i][j]);
        }
        printf("\n");
    }

    // evaluate hidden layer
    double *hidden_outputs = malloc((hidden->rows) * sizeof(double) + 1); // bias neuron in hidden layer
    matrix_dot(input, hidden, hidden_outputs);
    vec_apply(hidden_outputs, relu, hidden->rows + 1);
    hidden_outputs[hidden->rows] = 1; // bias neuron

    double *output_outputs = malloc(output->rows * sizeof(double)); // no bias neuron in output layer
    matrix_dot(hidden_outputs, output, output_outputs);
    softmax(output_outputs, output_outputs, output->rows);

    printf("Output: ");
    for (int i = 0; i < output->rows; i++) {
        printf("%.2f ", output_outputs[i]);
    }
    printf("\n");

    double error = 0;
    for (int i = 0; i < output->rows; i++) {
        error += labels[index] * log(output_outputs[i]);
    }

    printf("error: %.2f\n", error);

    
}

// general TODOs:
// 1. Osetrit mallocy
// 2. Skontrolovat ci sa niekde nepouziva int miesto double