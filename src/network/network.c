#include "network.h"

#define LR 0.1
#define HALVE_LR_AFTER_EPOCHS 10
#define LR_DECAY 0.95
#define EPOCHS 50
#define BATCH_SIZE 256
#define VALID_AFTER_EPOCHS 1
#define VALID_BATCH_SIZE 10000
#define VALID_SIZE 10000
#define MOMENTUM 0.9
#define MOMENTUM_DECAY 1

Network* network_create(int *layers, enum Activation *activations, int layer_count) {
    Network *network = (Network *) malloc(sizeof(Network));
    Matrix **layers_net = (Matrix **) malloc(layer_count * sizeof(Matrix*));

    for (int i = 0; i < layer_count; i++) {
        layers_net[i] = matrix_create(layers[i+1], layers[i] + 1); // +1 bias weight
    }

    network->activations = activations;
    network->layers = layers_net;
    network->layer_count = layer_count;

    return network;
}

// double compute_error(Matrix *matrix, Matrix *labels, double *out) {
//     double total_err = 0;
//     for (int i = 0; i < matrix->rows; i++) {
//         double err = 0;
//         for (int j = 0; j < matrix->cols; j++) {
//             double d_k = labels->data[i][0] == j ? 1 : 0;
//             err -= d_k * log(matrix->data[i][j]);
//         }
//         out[i] = err;
//         total_err += err;
//     }
//     return total_err;
// }

void network_predict(Network *network, Image **dataset, int dataset_size, char* predict_path) {
    FILE *out = fopen(predict_path, "wb");
    
    Matrix *predict_input = matrix_create(dataset_size, network->layers[0]->cols);
    for (int i = 0; i < dataset_size; i++) {
        int index = i;
        predict_input->data[i] = dataset[index]->data;
    }

    int layer_count = network->layer_count;
    Matrix **layers_transposed = (Matrix **) malloc(network->layer_count * sizeof(Matrix*));
    Matrix **layers_outputs = (Matrix **) malloc(network->layer_count * sizeof(Matrix*));
    Matrix **layers_activations = (Matrix **) malloc(network->layer_count * sizeof(Matrix*));

    for (int i = 0; i < network->layer_count; i++) {
        layers_transposed[i] = matrix_create(network->layers[i]->cols, network->layers[i]->rows);

        switch (network->activations[i]) {
            case RELU:
                layers_outputs[i] = matrix_create(dataset_size, network->layers[i]->rows + 1);
                layers_activations[i] = matrix_create(dataset_size, network->layers[i]->rows + 1);
                break;
            case SOFTMAX:
                layers_outputs[i] = matrix_create(dataset_size, network->layers[i]->rows);
                layers_activations[i] = matrix_create(dataset_size, network->layers[i]->rows);
                break;
        }
    }

    // forward pass
    for (int i = 0; i < network->layer_count; i++) {
        Matrix *input = i == 0 ? predict_input : layers_activations[i-1];
        matrix_transpose(network->layers[i], layers_transposed[i]);
        matrix_multiply(input, layers_transposed[i], layers_outputs[i]);
        switch (network->activations[i]) {
            case RELU:
                matrix_apply(layers_outputs[i], relu, layers_activations[i]);
                break;
            case SOFTMAX:
                matrix_apply_row_wise(layers_outputs[i], layers_activations[i], softmax);
                break;
        }

        if (i != network->layer_count - 1) {
            // set value of bias inputs for next layer to 1
            // no need to simulate bias input for output
            for (int j = 0; j < dataset_size; j++) {
                layers_activations[i]->data[j][network->layers[i]->rows] = 1;
            }
        }
    }

    for (int i = 0; i < dataset_size; i++) {
        int category = 0;
        double max_score = 0;
        for (int j = 0; j < network->layers[layer_count - 1]->rows; j++) {
            if (layers_activations[layer_count - 1]->data[i][j] > max_score) {
                category = j;
                max_score = layers_activations[layer_count - 1]->data[i][j];
            }
        }
        fprintf(out, "%d\n", category);
        fflush(out);
    }
    fclose(out);

    for (int i = 0; i < network->layer_count; i++) {
        matrix_free(layers_transposed[i]);
        matrix_free(layers_outputs[i]);
        matrix_free(layers_activations[i]);
    }
    free(layers_transposed);
    free(layers_outputs);
    free(layers_activations);
}

double network_validate(Network *network, Matrix *valid_input, Matrix *labels) {
    int layer_count = network->layer_count;
    Matrix **layers_transposed = (Matrix **) malloc(network->layer_count * sizeof(Matrix*));
    Matrix **layers_outputs = (Matrix **) malloc(network->layer_count * sizeof(Matrix*));
    Matrix **layers_activations = (Matrix **) malloc(network->layer_count * sizeof(Matrix*));

    for (int i = 0; i < network->layer_count; i++) {
        layers_transposed[i] = matrix_create(network->layers[i]->cols, network->layers[i]->rows);

        switch (network->activations[i]) {
            case RELU:
                layers_outputs[i] = matrix_create(VALID_BATCH_SIZE, network->layers[i]->rows + 1);
                layers_activations[i] = matrix_create(VALID_BATCH_SIZE, network->layers[i]->rows + 1);
                break;
            case SOFTMAX:
                layers_outputs[i] = matrix_create(VALID_BATCH_SIZE, network->layers[i]->rows);
                layers_activations[i] = matrix_create(VALID_BATCH_SIZE, network->layers[i]->rows);
                break;
        }
    }

    // forward pass
    for (int i = 0; i < network->layer_count; i++) {
        Matrix *input = i == 0 ? valid_input : layers_activations[i-1];
        matrix_transpose(network->layers[i], layers_transposed[i]);
        matrix_multiply(input, layers_transposed[i], layers_outputs[i]);
        switch (network->activations[i]) {
            case RELU:
                matrix_apply(layers_outputs[i], relu, layers_activations[i]);
                break;
            case SOFTMAX:
                matrix_apply_row_wise(layers_outputs[i], layers_activations[i], softmax);
                break;
        }

        if (i != network->layer_count - 1) {
            // set value of bias inputs for next layer to 1
            // no need to simulate bias input for output
            for (int j = 0; j < BATCH_SIZE; j++) {
                layers_activations[i]->data[j][network->layers[i]->rows] = 1;
            }
        }
    }

    int correct = 0;

    for (int i = 0; i < VALID_BATCH_SIZE; i++) {
        int category = 0;
        double max_score = 0;
        for (int j = 0; j < network->layers[layer_count - 1]->rows; j++) {
            if (layers_activations[layer_count - 1]->data[i][j] > max_score) {
                category = j;
                max_score = layers_activations[layer_count - 1]->data[i][j];
            }
        }
        if (labels->data[i][0] == category) {
            correct++;
        }
    }

    for (int i = 0; i < network->layer_count; i++) {
        matrix_free(layers_transposed[i]);
        matrix_free(layers_outputs[i]);
        matrix_free(layers_activations[i]);
    }
    free(layers_transposed);
    free(layers_outputs);
    free(layers_activations);


    return correct / ((double) VALID_BATCH_SIZE);
}

void network_train(Network *network, Network *best_model, Image **dataset, int dataset_size) {
    Matrix **layers_transposed = (Matrix **) malloc(network->layer_count * sizeof(Matrix*));
    Matrix **layers_outputs = (Matrix **) malloc(network->layer_count * sizeof(Matrix*));
    Matrix **layers_activations = (Matrix **) malloc(network->layer_count * sizeof(Matrix*));
    Matrix **layers_backprop_neurons = (Matrix **) malloc(network->layer_count * sizeof(Matrix*));
    Matrix **layers_backprop_weights = (Matrix **) malloc(network->layer_count * sizeof(Matrix*));
    Matrix **layers_momentum = (Matrix **) malloc(network->layer_count * sizeof(Matrix*));

    // layer init
    for (int i = 0; i < network->layer_count; i++) {
        layers_transposed[i] = matrix_create(network->layers[i]->cols, network->layers[i]->rows);
        layers_backprop_neurons[i] = matrix_create(BATCH_SIZE, network->layers[i]->rows);
        layers_backprop_weights[i] = matrix_create(network->layers[i]->rows, network->layers[i]->cols);
        layers_momentum[i] = matrix_create(network->layers[i]->rows, network->layers[i]->cols);

        switch (network->activations[i]) {
            case RELU:
                he_init(network->layers[i], network->layers[i]->cols); // He init for layers with ReLU
                layers_outputs[i] = matrix_create(BATCH_SIZE, network->layers[i]->rows + 1);
                layers_activations[i] = matrix_create(BATCH_SIZE, network->layers[i]->rows + 1);
                break;
            case SOFTMAX:
                glorot_init(network->layers[i], network->layers[i]->cols, network->layers[i]->rows); // Uniform Glorot init for regular layers
                layers_outputs[i] = matrix_create(BATCH_SIZE, network->layers[i]->rows);
                layers_activations[i] = matrix_create(BATCH_SIZE, network->layers[i]->rows);
                break;
        }
    }

    // double *error_vec = malloc(BATCH_SIZE * sizeof(double));

    Matrix *input_batch = matrix_create(BATCH_SIZE, network->layers[0]->cols);
    Matrix *label_batch = matrix_create(BATCH_SIZE, 1);

    Matrix *inputs_valid = matrix_create(VALID_SIZE, network->layers[0]->cols);
    Matrix *labels_valid = matrix_create(VALID_SIZE, 1);

    int last_used_image = 0;
    int last_used_valid = 0;
    double best_valid_score = 0;

    for (int epoch = 0; epoch < EPOCHS; epoch++){
        for (int batch = 0; batch < dataset_size / BATCH_SIZE; batch++) {
            if (batch % 4 == 0) {
                printf("-");
                fflush(stdout);
            }

            // setting array of weights in backprop to 0, other arrays are overwritten
            for (int i = 0; i < network->layer_count; i++) {
                matrix_set(layers_backprop_weights[i], 0);
            }

            // batch init
            for (int i = 0; i < BATCH_SIZE; i++) { // SGD
                int index = (i + last_used_image) % (dataset_size - VALID_SIZE);
                input_batch->data[i] = dataset[index]->data;
                label_batch->data[i][0] = dataset[index]->label;
            }
            last_used_image = (last_used_image + BATCH_SIZE) % dataset_size;

            // forward pass
            for (int i = 0; i < network->layer_count; i++) {
                Matrix *input = i == 0 ? input_batch : layers_activations[i-1];
                matrix_transpose(network->layers[i], layers_transposed[i]);
                matrix_multiply(input, layers_transposed[i], layers_outputs[i]);
                switch (network->activations[i]) {
                    case RELU:
                        matrix_apply(layers_outputs[i], relu, layers_activations[i]);
                        break;
                    case SOFTMAX:
                        matrix_apply_row_wise(layers_outputs[i], layers_activations[i], softmax);
                        break;
                }

                if (i != network->layer_count - 1) {
                    // set value of bias inputs for next layer to 1
                    // no need to simulate bias input for output
                    for (int j = 0; j < BATCH_SIZE; j++) {
                        layers_activations[i]->data[j][network->layers[i]->rows] = 1;
                    }
                }
            }

            // actual values of error functions are not needed, they are here just for graphing

            // double error = compute_error(layers_activations[network->layer_count - 1], label_batch, error_vec);
            // printf("epoch: %d, batch: %d, error: %.20f\n", epoch, batch, error/BATCH_SIZE);
            // fprintf(fd_err, "%.10f\n", error/BATCH_SIZE);
            // fflush(fd_err);

            // GD - backprop
            for (int layer = network->layer_count - 1; layer >= 0; layer--) {
                Matrix *previous_layer = (layer == 0) ? input_batch : layers_activations[layer - 1];
                switch (network->activations[layer]) {
                    // error gradient for RELU layers
                    case RELU:
                        // calculate error gradient for neurons
                        #pragma omp parallel for collapse(2) num_threads(16)
                        for (int i = 0; i < BATCH_SIZE; i++) {
                            for (int j = 0; j < network->layers[layer]->rows; j++) {
                                layers_backprop_neurons[layer]->data[i][j] = 0;
                                for (int k = 0; k < network->layers[layer+1]->rows; k++) {
                                    // sacrificed atomicity for speed :(
                                    // #pragma omp atomic
                                    layers_backprop_neurons[layer]->data[i][j] += layers_backprop_neurons[layer+1]->data[i][k] * network->layers[layer+1]->data[k][j];
                                }
                            }
                        }

                        // calculate error gradient for weights
                        #pragma omp parallel for collapse(2) num_threads(16)
                        for (int i = 0; i < BATCH_SIZE; i++) {
                            for (int j = 0; j < network->layers[layer]->rows; j++) {
                                for (int k = 0; k < network->layers[layer]->cols; k++) {
                                    // #pragma omp atomic
                                    layers_backprop_weights[layer]->data[j][k] += layers_backprop_neurons[layer]->data[i][j] * (layers_outputs[layer]->data[i][j] > 0) * previous_layer->data[i][k];
                                }
                            }
                        }
                        break;
                    
                    // error gradient for softmax layers
                    case SOFTMAX:
                        // here it is expected that the softmax is also the last layer
                        
                        // calculate error gradient for neurons
                        for (int i = 0; i < BATCH_SIZE; i++) {
                            for (int j = 0; j < network->layers[layer]->rows; j++) {
                                layers_backprop_neurons[layer]->data[i][j] = layers_activations[layer]->data[i][j] - (label_batch->data[i][0] == j ? 1 : 0); //SM_i - d_i
                            }
                        }

                        // calculate error gradient for weights
                        #pragma omp parallel for collapse(2) num_threads(16)
                        for (int i = 0; i < BATCH_SIZE; i++) {
                            for (int j = 0; j < network->layers[layer]->rows; j++) {
                                for (int k = 0; k < network->layers[layer]->cols; k++) {
                                    // #pragma omp atomic
                                    layers_backprop_weights[layer]->data[j][k] += layers_backprop_neurons[layer]->data[i][j] * previous_layer->data[i][k];
                                }
                            }
                        }
                        break;
                }
            }

            // LR computation
            double constant = -LR * pow(LR_DECAY, epoch) * (1/((float) BATCH_SIZE));

            // one step of gradient descent - multiply gradient by -LR, add momentum and add to current weights
            for (int i = 0; i < network->layer_count; i++) {
                matrix_multiply_by_constant(layers_backprop_weights[i], constant);

                // compute momentum
                if (epoch != 0 && batch != 0) {
                    matrix_add(layers_backprop_weights[i], layers_momentum[i], layers_backprop_weights[i]);
                }
                matrix_copy(layers_backprop_weights[i], layers_momentum[i]);
                matrix_multiply_by_constant(layers_momentum[i], MOMENTUM * pow(MOMENTUM_DECAY, epoch));
                matrix_add(network->layers[i], layers_backprop_weights[i], network->layers[i]);
            }
        }
        printf("\n");

        // validation
        if (epoch % VALID_AFTER_EPOCHS == 0) {
            for (int i = 0; i < VALID_SIZE; i++) {
                int index = (i + last_used_valid) % (VALID_SIZE) + (dataset_size - VALID_SIZE);
                inputs_valid->data[i] = dataset[index]->data;
                labels_valid->data[i][0] = dataset[index]->label;
            }
            last_used_valid = (last_used_valid + VALID_SIZE) % dataset_size;

            double valid_score = network_validate(network, inputs_valid, labels_valid);
            printf("Epoch %d, LR: %f, Momentum: %f, Valid score: %f\n", epoch, LR * pow(LR_DECAY, epoch), MOMENTUM * pow(MOMENTUM_DECAY, epoch), valid_score);

            if (valid_score > best_valid_score) {
                printf("Found new best model with valid accuracy: %f\n", valid_score);
                best_valid_score = valid_score;
                for (int i = 0; i < network->layer_count; i++) {
                    matrix_copy(network->layers[i], best_model->layers[i]);
                }
            }
        }
    }

    for (int i = 0; i < network->layer_count; i++) {
        matrix_free(layers_transposed[i]);
        matrix_free(layers_outputs[i]);
        matrix_free(layers_activations[i]);
        matrix_free(layers_backprop_neurons[i]);
        matrix_free(layers_backprop_weights[i]);
        matrix_free(layers_momentum[i]);
    }
    free(layers_transposed);
    free(layers_outputs);
    free(layers_activations);
    free(layers_backprop_neurons);
    free(layers_backprop_weights);
    free(layers_momentum);
}