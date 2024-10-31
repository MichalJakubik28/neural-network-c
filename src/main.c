#include <stdio.h>
#include <math.h>
#include "utils/csv_to_img.h"
// #include "matrix/matrix.h"
#include "matrix/vec_ops.h"
#include "weight_ops/weight_ops.h"

#ifndef IMG_NUM
#define IMG_NUM 0
#endif

#define LR 0.01
#define INPUT 2
#define HIDDEN 8
#define OUTPUT 2
#define EPOCHS 100000
#define BATCH_SIZE 10

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
    double labels[4] = {0, 1, 1, 0};

    // hidden layer
    Matrix *hidden = matrix_create(HIDDEN, INPUT + 1); // +1 so each neuron has a bias
    he_init(hidden, INPUT + 1); // He init for layers with ReLU
    printf("Initialized weights in hidden layer:\n");
    for (int i = 0; i < hidden->rows; i++) {
        for (int j = 0; j < hidden->cols; j++) {
            printf("%.2f ", hidden->data[i][j]);
        }
        printf("\n");
    }

    // output layer
    Matrix *output = matrix_create(OUTPUT, HIDDEN + 1); // +1 so each neuron has a bias
    glorot_init(output, HIDDEN, OUTPUT); // Uniform Glorot init for regular layers 
    printf("Initialized weights in output layer:\n");
    for (int i = 0; i < output->rows; i++) {
        for (int j = 0; j < output->cols; j++) {
            printf("%.2f ", output->data[i][j]);
        }
        printf("\n");
    }


    // loop start
    double *hidden_outputs = malloc((hidden->rows + 1) * sizeof(double)); // bias neuron in hidden layer
    double *hidden_outputs_relu = malloc((hidden->rows + 1) * sizeof(double));
    double *output_outputs = malloc(output->rows * sizeof(double)); // no bias neuron in output layer
    double *output_softmax = malloc(output->rows * sizeof(double));
    double *output_backprop_neuron = malloc(output->rows * sizeof(double));
    Matrix *output_backprop_weight = matrix_create(output->rows, output->cols);
    double *hidden_backprop_neuron = malloc(hidden->rows * sizeof(double));
    Matrix *hidden_backprop_weight = matrix_create(hidden->rows, hidden->cols);


    for (int epoch = 0; epoch < EPOCHS; epoch++){
        // if (epoch == 0 || epoch == EPOCHS - 1) {
        //     printf("Weights in hidden layer:\n");
        //     for (int i = 0; i < hidden->rows; i++) {
        //         for (int j = 0; j < hidden->cols; j++) {
        //             printf("%.2f ", hidden->data[i][j]);
        //         }
        //         printf("\n");
        //     }
        //     printf("Weights in output layer:\n");
        //     for (int i = 0; i < output->rows; i++) {
        //         for (int j = 0; j < output->cols; j++) {
        //             printf("%.2f ", output->data[i][j]);
        //         }
        //         printf("\n");
        //     }
        // }
        memset(hidden_outputs, 0, (hidden->rows + 1) * sizeof(double));
        memset(hidden_outputs_relu, 0, (hidden->rows + 1) * sizeof(double));
        memset(output_outputs, 0, output->rows * sizeof(double));
        memset(output_softmax, 0, output->rows * sizeof(double));
        memset(output_backprop_neuron, 0, output->rows * sizeof(double));
        matrix_set(output_backprop_weight, 0);
        memset(hidden_backprop_neuron, 0, hidden->rows * sizeof(double));
        matrix_set(hidden_backprop_weight, 0);

        int index = rand() % 4;
        double *input = inputs[index];


        // FORWARD PASS
        // evaluate hidden layer
        matrix_dot(input, hidden, hidden_outputs);
        vec_apply(hidden_outputs, relu, hidden_outputs_relu, hidden->rows + 1);
        hidden_outputs_relu[hidden->rows] = 1; // bias neuron

        // evaluate output layer
        matrix_dot(hidden_outputs_relu, output, output_outputs);
        softmax(output_outputs, output_softmax, output->rows);

        printf("Output: ");
        for (int i = 0; i < output->rows; i++) {
            printf("%f ", output_softmax[i]);
        }
        printf("\n");

        double error = 0;
        for (int i = 0; i < output->rows; i++) {
            error -= (labels[index] != i ? 1 : 0) * log(output_softmax[i]);
        }

        printf("error: %f\n", error);


        // calculate error gradient for output neurons
        for (int i = 0; i < output->rows; i++) {
            output_backprop_neuron[i] = output_softmax[i] - (labels[index] != i ? 1 : 0); //SM_i - d_i
        }


        // calculate error gradient for output weights
        for (int i = 0; i < output->rows; i++) {
            for (int j = 0; j < output->cols; j++) {
                output_backprop_weight->data[i][j] = output_backprop_neuron[i] * hidden_outputs_relu[j];
            }
        }

        // calculate error gradient for hidden neurons
        for (int i = 0; i < hidden->rows; i++) {
            hidden_backprop_neuron[i] = 0;
            for (int j = 0; j < output->rows; j++) {
                hidden_backprop_neuron[i] += output_backprop_neuron[j] * output->data[j][i];
            }
        }


        // calculate error gradient for hidden weights
        for (int i = 0; i < hidden->rows; i++) {
            for (int j = 0; j < hidden->cols; j++) {
                hidden_backprop_weight->data[i][j] = hidden_backprop_neuron[i] * relu_prime(hidden_outputs[i]) * input[j];
            }
        }

        matrix_multiply_by_constant(output_backprop_weight, -LR);
        matrix_multiply_by_constant(hidden_backprop_weight, -LR);

        // one step of gradient descent
        matrix_add(output, output_backprop_weight, output);
        matrix_add(hidden, hidden_backprop_weight, hidden);
    }
}

// general TODOs:
// 1. Osetrit mallocy
// 2. Skontrolovat ci sa niekde nepouziva int miesto double
// 3. ucia sa aj biasy?
// 4. 