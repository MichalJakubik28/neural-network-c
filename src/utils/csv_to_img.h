#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    double *data;
    int label;
    int prediction;
} Image;

void free_dataset(Image **dataset, int size);
Image** csv_to_imgs(char *path, int img_size, int *dataset_size);
void parse_labels(char *path, Image **dataset, int dataset_size);
void shuffle_dataset(Image **dataset, int dataset_size);