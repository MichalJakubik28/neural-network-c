#include <stdio.h>
#include <math.h>
#include "utils/csv_to_img.h"
#include "matrix/vec_ops.h"
#include "weight_ops/weight_ops.h"
#include <time.h>

#define LR 0.1
#define HALVE_LR_AFTER_EPOCHS 10
#define LR_DECAY 0.925
#define INPUT 784
#define HIDDEN 512
#define HIDDEN2 256
#define OUTPUT 10
#define EPOCHS 20000
#define BATCH_SIZE 256
#define VALID_AFTER 100
#define VALID_BATCH_SIZE 10000
#define VALID_SIZE 10000
#define MOMENTUM 0.9

double compute_error(Matrix *matrix, Matrix *labels, double *out) {
    double total_err = 0;
    for (int i = 0; i < matrix->rows; i++) {
        double err = 0;
        for (int j = 0; j < matrix->cols; j++) {
            double d_k = labels->data[i][0] == j ? 1 : 0;
            err -= d_k * log(matrix->data[i][j]);
        }
        out[i] = err;
        total_err += err;
    }
    return total_err;
}

double validation_epoch(Matrix *input, Matrix *hidden, Matrix *hidden2, Matrix *output, Matrix *labels) {
    Matrix *hidden_out = matrix_create(VALID_BATCH_SIZE, hidden->rows + 1);
    Matrix *hidden_out_relu = matrix_create(VALID_BATCH_SIZE, hidden->rows + 1);
    Matrix *hidden2_out = matrix_create(VALID_BATCH_SIZE, hidden2->rows + 1);
    Matrix *hidden2_out_relu = matrix_create(VALID_BATCH_SIZE, hidden2->rows + 1);
    Matrix *output_out = matrix_create(VALID_BATCH_SIZE, output->rows);
    Matrix *output_sm = matrix_create(VALID_BATCH_SIZE, output->rows);

    Matrix *hidden_transposed = matrix_create(hidden->cols, hidden->rows);
    Matrix *hidden2_transposed = matrix_create(hidden2->cols, hidden2->rows);
    Matrix *output_transposed = matrix_create(output->cols, output->rows);

    // FORWARD PASS
    // evaluate hidden layer
    matrix_transpose(hidden, hidden_transposed);
    matrix_multiply(input, hidden_transposed, hidden_out);
    matrix_apply(hidden_out, relu, hidden_out_relu);

    // set value of bias neurons for next layer to 1 
    for (int i = 0; i < VALID_BATCH_SIZE; i++) {
        hidden_out_relu->data[i][hidden->rows] = 1;
    }

    // evaluate second hidden layer
    matrix_transpose(hidden2, hidden2_transposed);
    matrix_multiply(hidden_out_relu, hidden2_transposed, hidden2_out);
    matrix_apply(hidden2_out, relu, hidden2_out_relu);

    // set value of bias neurons for next layer to 1 
    for (int i = 0; i < VALID_BATCH_SIZE; i++) {
        hidden2_out_relu->data[i][hidden2->rows] = 1;
    }

    matrix_transpose(output, output_transposed);
    matrix_multiply(hidden2_out_relu, output_transposed, output_out);
    
    matrix_apply_row_wise(output_out, output_sm, softmax);
    int correct = 0;

    for (int i = 0; i < VALID_BATCH_SIZE; i++) {
        int category = 0;
        double max_score = 0;
        for (int j = 0; j < output->rows; j++) {
            if (output_sm->data[i][j] > max_score) {
                category = j;
                max_score = output_sm->data[i][j];
            }
        }
        if (labels->data[i][0] == category) {
            correct++;
        }
    }

    matrix_free(hidden_out);
    matrix_free(hidden_out_relu);
    matrix_free(hidden2_out);
    matrix_free(hidden2_out_relu);
    matrix_free(output_out);
    matrix_free(output_sm);
    matrix_free(hidden_transposed);
    matrix_free(hidden2_transposed);
    matrix_free(output_transposed);

    return correct / ((double) VALID_BATCH_SIZE);
}

int main() {
    FILE *fd_err = fopen("../out/error_values.txt", "wb");
    FILE *fd_valid = fopen("../out/valid_values.txt", "wb");

    // srand(42);
    srand(time(NULL));
    int dataset_size;
    Image **dataset = csv_to_imgs("../data/fashion_mnist_train_vectors.csv", 28, &dataset_size);
    parse_labels("../data/fashion_mnist_train_labels.csv", dataset, dataset_size);
    // int image_num = (int) (rand()/RAND_MAX * 9);
    // int image_num = 59999;
    // print_matrix(dataset[image_num]->data, 28);
    // printf("Label: %d\n", dataset[image_num]->label);

    shuffle_dataset(dataset, dataset_size);

    // hidden layer
    Matrix *hidden = matrix_create(HIDDEN, INPUT + 1); // +1 so each neuron has a bias
    he_init(hidden, INPUT + 1); // He init for layers with ReLU
    printf("Initialized weights in hidden layer:\n");
    matrix_print(hidden);

    // second hidden layer
    Matrix *hidden2 = matrix_create(HIDDEN2, HIDDEN + 1); // +1 so each neuron has a bias
    he_init(hidden2, INPUT + 1); // He init for layers with ReLU
    printf("Initialized weights in second hidden layer:\n");
    matrix_print(hidden2);

    // output layer
    Matrix *output = matrix_create(OUTPUT, HIDDEN2 + 1); // +1 so each neuron has a bias
    glorot_init(output, HIDDEN, OUTPUT); // Uniform Glorot init for regular layers 
    printf("Initialized weights in output layer:\n");
    matrix_print(output);

    // loop start
    Matrix *hidden_out = matrix_create(BATCH_SIZE, hidden->rows + 1);
    Matrix *hidden_out_relu = matrix_create(BATCH_SIZE, hidden->rows + 1);
    Matrix *hidden2_out = matrix_create(BATCH_SIZE, hidden2->rows + 1);
    Matrix *hidden2_out_relu = matrix_create(BATCH_SIZE, hidden2->rows + 1);
    Matrix *output_out = matrix_create(BATCH_SIZE, output->rows);
    Matrix *output_sm = matrix_create(BATCH_SIZE, output->rows);
    Matrix *output_bp_neuron = matrix_create(BATCH_SIZE, output->rows);
    Matrix *output_backprop_weight = matrix_create(output->rows, output->cols);
    Matrix *hidden2_bp_neuron = matrix_create(BATCH_SIZE, hidden2->rows);
    Matrix *hidden2_backprop_weight = matrix_create(hidden2->rows, hidden2->cols);
    Matrix *hidden_bp_neuron = matrix_create(BATCH_SIZE, hidden->rows);
    Matrix *hidden_backprop_weight = matrix_create(hidden->rows, hidden->cols);

    Matrix *output_momentum = matrix_create(output->rows, output->cols);
    Matrix *hidden2_momentum = matrix_create(hidden2->rows, hidden2->cols);
    Matrix *hidden_momentum = matrix_create(hidden->rows, hidden->cols);

    Matrix *hidden_transposed = matrix_create(hidden->cols, hidden->rows);
    Matrix *hidden2_transposed = matrix_create(hidden2->cols, hidden2->rows);
    Matrix *output_transposed = matrix_create(output->cols, output->rows);

    double *error_vec = malloc(BATCH_SIZE * sizeof(double));

    Matrix *input_batch = matrix_create(BATCH_SIZE, INPUT + 1);
    Matrix *label_batch = matrix_create(BATCH_SIZE, 1);

    Matrix *inputs_valid = matrix_create(VALID_SIZE, INPUT + 1);
    Matrix *labels_valid = matrix_create(VALID_SIZE, 1);

    int last_used_image = 0;
    int last_used_valid = 0;

    for (int epoch = 0; epoch < EPOCHS; epoch++){
        // if (epoch == 0 || epoch == EPOCHS - 1) {
        //     printf("Weights after epoch %d:\n", epoch);
        //     printf("Weights in hidden layer:\n");
        //     matrix_print(hidden);
        //     printf("Weights in output layer:\n");
        //     matrix_print(output);
        // }
        matrix_set(output_backprop_weight, 0);
        matrix_set(hidden_backprop_weight, 0);

        matrix_set(hidden_out, 0);
        matrix_set(hidden_out_relu, 0);
        matrix_set(hidden_bp_neuron, 0);
        matrix_set(hidden2_out, 0);
        matrix_set(hidden2_out_relu, 0);
        matrix_set(hidden2_bp_neuron, 0);        
        matrix_set(output_bp_neuron, 0);
        matrix_set(output_out, 0);
        matrix_set(output_sm, 0);
        memset(error_vec, 0, BATCH_SIZE * sizeof(double));
        // print_matrix(dataset[0]->data, 28);

        for (int i = 0; i < BATCH_SIZE; i++) { // SGD
            int index = (i + last_used_image) % (dataset_size - VALID_SIZE);
            input_batch->data[i] = dataset[index]->data;
            label_batch->data[i][0] = dataset[index]->label;
        }
        last_used_image = (last_used_image + BATCH_SIZE) % dataset_size;

        // FORWARD PASS
        // evaluate hidden layer
        matrix_transpose(hidden, hidden_transposed);
        matrix_multiply(input_batch, hidden_transposed, hidden_out);
        matrix_apply(hidden_out, relu, hidden_out_relu);

        // set value of bias neurons for next layer to 1 
        for (int i = 0; i < BATCH_SIZE; i++) {
            hidden_out_relu->data[i][hidden->rows] = 1;
        }

        // evaluate second hidden layer
        matrix_transpose(hidden2, hidden2_transposed);
        matrix_multiply(hidden_out_relu, hidden2_transposed, hidden2_out);
        matrix_apply(hidden2_out, relu, hidden2_out_relu);

        // set value of bias neurons for next layer to 1 
        for (int i = 0; i < BATCH_SIZE; i++) {
            hidden2_out_relu->data[i][hidden2->rows] = 1;
        }

        matrix_transpose(output, output_transposed);
        matrix_multiply(hidden2_out_relu, output_transposed, output_out);
        
        matrix_apply_row_wise(output_out, output_sm, softmax);

        double error = compute_error(output_sm, label_batch, error_vec);        

        // printf("Output: ");
        // for (int i = 0; i < output->rows; i++) {
        //     printf("%.20f ", output_sm->data[0][i]);
        // }
        // printf("\n");

        printf("epoch: %d, error: %.20f\n", epoch, error/BATCH_SIZE);
        fprintf(fd_err, "%.10f\n", error/BATCH_SIZE);
        fflush(fd_err);

        // printf("Total error: %f, inidividual errors: ", error);
        // for (int i = 0; i < BATCH_SIZE; i++) {
        //     printf("%f ", error_vec[i]);
        // }
        // printf("\n");

        // calculate error gradient for output neurons
        for (int i = 0; i < BATCH_SIZE; i++) {
            for (int j = 0; j < output->rows; j++) {
                output_bp_neuron->data[i][j] = output_sm->data[i][j] - (label_batch->data[i][0] == j ? 1 : 0); //SM_i - d_i
            }
        }
        // printf("Gradient of output neurons:\n");
        // matrix_print(output_bp_neuron);

        // calculate error gradient for output weights
        #pragma omp parallel for collapse(2) shared(output, output_backprop_weight)
        for (int i = 0; i < BATCH_SIZE; i++) {
            for (int j = 0; j < output->rows; j++) {
                for (int k = 0; k < output->cols; k++) {
                    output_backprop_weight->data[j][k] += output_bp_neuron->data[i][j] * hidden2_out_relu->data[i][k];
                }
            }
        }
        // printf("Gradient of output weights:\n");
        // matrix_print(output_backprop_weight);

        // calculate error gradient for second hidden neurons
        #pragma omp parallel for collapse(2) shared(hidden2, hidden2_bp_neuron)
        for (int i = 0; i < BATCH_SIZE; i++) {
            for (int j = 0; j < hidden2->rows; j++) {
                hidden2_bp_neuron->data[i][j] = 0;
                for (int k = 0; k < output->rows; k++) {
                    hidden2_bp_neuron->data[i][j] += output_bp_neuron->data[i][k] * output->data[k][j];
                }
            }
        }
        // printf("Gradient of second hidden neurons:\n");
        // matrix_print(hidden_bp_neuron);

        // calculate error gradient for second hidden weights
        double tmp;
        #pragma omp parallel for collapse(2) shared(hidden2, tmp, hidden2_backprop_weight)
        for (int i = 0; i < BATCH_SIZE; i++) {
            for (int j = 0; j < hidden2->rows; j++) {
                for (int k = 0; k < hidden2->cols; k++) {
                    tmp = hidden2_bp_neuron->data[i][j] * (hidden2_out->data[i][j] > 0) * hidden_out_relu->data[i][k];
                    hidden2_backprop_weight->data[j][k] += tmp;
                }
            }
        }

        // calculate error gradient for hidden neurons
        #pragma omp parallel for collapse(2) shared(hidden, hidden_bp_neuron)
        for (int i = 0; i < BATCH_SIZE; i++) {
            for (int j = 0; j < hidden->rows; j++) {
                hidden_bp_neuron->data[i][j] = 0;
                for (int k = 0; k < hidden2->rows; k++) {
                    hidden_bp_neuron->data[i][j] += hidden2_bp_neuron->data[i][k] * hidden2->data[k][j];
                }
            }
        }
        // printf("Gradient of hidden neurons:\n");
        // matrix_print(hidden_bp_neuron);

        // calculate error gradient for hidden weights
        #pragma omp parallel for collapse(2) shared(hidden, hidden_backprop_weight)
        for (int i = 0; i < BATCH_SIZE; i++) {
            for (int j = 0; j < hidden->rows; j++) {
                for (int k = 0; k < hidden->cols; k++) {
                    hidden_backprop_weight->data[j][k] += hidden_bp_neuron->data[i][j] * (hidden_out->data[i][j] > 0) * input_batch->data[i][k];
                }
            }
        }
        // printf("Gradient of hidden weights:\n");
        // matrix_print(hidden_backprop_weight);

        // double constant = -(LR/(pow(2, ((epoch * BATCH_SIZE)/ (dataset_size * HALVE_LR_AFTER_EPOCHS))))) * (1/((float) BATCH_SIZE));
        double constant = -LR * pow(LR_DECAY, (epoch * BATCH_SIZE)/ (dataset_size)) * (1/((float) BATCH_SIZE));
        // printf("LR: %f ", (LR/(pow(2, ((epoch * BATCH_SIZE)/ (dataset_size * HALVE_LR_AFTER_EPOCHS))))));
        printf("LR: %f ", LR * pow(LR_DECAY, (epoch * BATCH_SIZE)/ (dataset_size)));
        matrix_multiply_by_constant(output_backprop_weight, constant);
        matrix_multiply_by_constant(hidden2_backprop_weight, constant);
        matrix_multiply_by_constant(hidden_backprop_weight, constant);
        // printf("Final output weight vector:\n");
        // matrix_print(output_backprop_weight);
        // printf("Final hidden weight vector:\n");
        // matrix_print(hidden_backprop_weight);


        if (epoch != 0) {
            matrix_add(output_backprop_weight, output_momentum, output_backprop_weight);
            matrix_add(hidden2_backprop_weight, hidden2_momentum, hidden2_backprop_weight);
            matrix_add(hidden_backprop_weight, hidden_momentum, hidden_backprop_weight);
        }

        matrix_copy(output_backprop_weight, output_momentum);
        matrix_copy(hidden2_backprop_weight, hidden2_momentum);
        matrix_copy(hidden_backprop_weight, hidden_momentum);

        matrix_multiply_by_constant(output_momentum, MOMENTUM);
        matrix_multiply_by_constant(hidden2_momentum, MOMENTUM);
        matrix_multiply_by_constant(hidden_momentum, MOMENTUM);

        // one step of gradient descent
        matrix_add(output, output_backprop_weight, output);
        matrix_add(hidden2, hidden2_backprop_weight, hidden2);
        matrix_add(hidden, hidden_backprop_weight, hidden);

        if (epoch % VALID_AFTER == 0) {
            for (int i = 0; i < VALID_SIZE; i++) {
                int index = (i + last_used_valid) % (VALID_SIZE) + (dataset_size - VALID_SIZE);
                inputs_valid->data[i] = dataset[index]->data;
                labels_valid->data[i][0] = dataset[index]->label;
            }
            last_used_valid = (last_used_valid + VALID_SIZE) % dataset_size;

            double valid_score = validation_epoch(inputs_valid, hidden, hidden2, output, labels_valid);
            printf("VALID SCORE: %f\n", valid_score);
            fprintf(fd_valid, "%.10f\n", valid_score);
            fflush(fd_valid);
        }
    }

    fclose(fd_err);
    fclose(fd_valid);
    free_dataset(dataset, dataset_size);

}

// general TODOs:
// 1. Osetrit mallocy
// 2. Skontrolovat ci sa niekde nepouziva int miesto double
// 3. ucia sa aj biasy?
// 4. skontrolovat ci sa vsetko nuluje