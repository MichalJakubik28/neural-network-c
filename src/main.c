#include "utils/csv_to_img.c"
#include "matrix/matrix.c"

#ifndef IMG_NUM
#define IMG_NUM 0
#endif

int main() {
    int dataset_size;
    Image **dataset = csv_to_imgs("../data/fashion_mnist_train_vectors.csv", 28, &dataset_size);
    print_matrix(dataset[IMG_NUM]->data, 28);
    free_dataset(dataset, dataset_size);
    //TODO uvolnit dataset
}