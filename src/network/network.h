#ifndef MATRIX_H
#define MATRIX_H
#include "../matrix/matrix.h"
#endif
#ifndef CSV_TO_IMG_H
#define CSV_TO_IMG_H
#include "../utils/csv_to_img.h"
#endif
#ifndef WEIGHT_OPS_H
#define WEIGHT_OPS_H
#include "../weight_ops/weight_ops.h"
#endif

enum Activation {
    RELU,
    SOFTMAX
};

typedef struct {
    Matrix **layers;
    enum Activation *activations;
    int layer_count;
} Network;

void network_train(Network *network, Network *best_model, Image **dataset, int dataset_size);
double network_validate(Network *network, Matrix *valid_input, Matrix *labels);
void network_predict(Network *network, Image **dataset, int dataset_size, char* predict_path);
Network* network_create(int *layers, enum Activation *activations, int layer_count);
