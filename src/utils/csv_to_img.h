#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    double *data;
    int label;
    int prediction;
} Image;

Image** csv_to_imgs(char *path, int img_size, int *dataset_size);