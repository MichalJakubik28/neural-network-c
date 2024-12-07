#include <stdio.h>
#include <math.h>
#ifndef CSV_TO_IMG_H
#define CSV_TO_IMG_H
#include "utils/csv_to_img.h"
#endif
#ifndef WEIGHT_OPS_H
#define WEIGHT_OPS_H
#include "weight_ops/weight_ops.h"
#endif
#include <time.h>
#include "network/network.h"

int main() {
    srand(time(NULL));
    printf("Parsing training dataset");
    fflush(stdout);
    int dataset_size;
    Image **train_dataset = csv_to_imgs("data/fashion_mnist_train_vectors.csv", 28, &dataset_size);
    parse_labels("data/fashion_mnist_train_labels.csv", train_dataset, dataset_size);

    Image **train_dataset_shuffled = malloc(dataset_size*sizeof(Image*));
    shallow_copy_dataset(train_dataset, train_dataset_shuffled, dataset_size);
    shuffle_dataset(train_dataset_shuffled, dataset_size);
    printf(" - OK\n");
    fflush(stdout);

    int layers[] = {784, 256, 10};
    enum Activation activations[] = {RELU, SOFTMAX};

    Network *network = network_create(layers, activations, 2);
    Network *best_model = network_create(layers, activations, 2);

    printf("Training...\n\n");
    fflush(stdout);
    network_train(network, best_model, train_dataset_shuffled, dataset_size);

    printf("\nPredicting training dataset");
    fflush(stdout);
    network_predict(best_model, train_dataset, dataset_size, "train_predictions.csv");
    free_dataset(train_dataset, dataset_size);
    printf(" - OK\n");
    fflush(stdout);

    printf("Parsing testing dataset\n");
    fflush(stdout);
    Image **test_dataset = csv_to_imgs("data/fashion_mnist_test_vectors.csv", 28, &dataset_size);

    printf("Predicting training dataset");
    fflush(stdout);
    network_predict(best_model, test_dataset, dataset_size, "test_predictions.csv");
    free_dataset(test_dataset, dataset_size);
    printf(" - OK\n");
    printf("Finished.\n");
    fflush(stdout);
}

// Known issues:
// 1. Some memory is not freed
// 2. Some variables should be updated atomically but are not because of speed
