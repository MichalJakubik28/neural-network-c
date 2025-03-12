#include "csv_to_img.h"
#include <unistd.h>

#define LINE_MAX_SIZE img_size * img_size * 4

bool add_to_dataset(Image *image, Image ***dataset, int *dataset_max_size, int *dataset_cur_size) {
    if (*dataset_cur_size == *dataset_max_size) {
        Image **new_dataset = realloc(*dataset, (*dataset_max_size) * 2 * sizeof(double));
        if (new_dataset == NULL) {
            return false;
        }
        *dataset = new_dataset;
        *dataset_max_size *= 2;
    }

    (*dataset)[*dataset_cur_size] = image;
    *dataset_cur_size += 1;
    return true;
}

Image* parse_line(char *line, int img_size) {
    Image *img = malloc(sizeof(Image));
    img->label = -1;
    img->prediction = -1;
    double *matrix = malloc((img_size * img_size + 1) * sizeof(double));

    double cur_num = 0;
    int num_total = 0;
    char *num_str = strtok(line, ",");
    while (num_str != NULL) {
        sscanf(num_str, "%lf", &cur_num);
        matrix[num_total] = cur_num / ((double) 255); // Normalize to [0,1]
        num_total++;
        num_str = strtok(NULL, ",");
    }

    matrix[img_size * img_size] = 1; // bias neuron
    img->data = matrix;
    return img;
}

void free_image(Image *img) {
    free(img->data);
    free(img);
}

void free_dataset(Image **dataset, int size) {
    for (int i = 0; i < size; i++) {
        free_image(dataset[i]);
    }
    free(dataset);
}

Image** csv_to_imgs(char *path, int img_size, int *dataset_size) {
    FILE *fd = fopen(path, "r");
    int dataset_max_size = 128;
    int dataset_cur_size = 0;
    Image **dataset = malloc(dataset_max_size*sizeof(Image*));
    if (dataset == NULL) {
        perror("Could not allocate enough memory for new dataset");
        fclose(fd);
    }
    int cur_char;
    char *line = malloc(LINE_MAX_SIZE * sizeof(char));
    if (line == NULL) {
        perror("Could not allocate enough memory for new image");
        free(dataset);
        fclose(fd);
    }
    int read_chars = 0;

    while ((cur_char = fgetc(fd)) != EOF) {
        if (cur_char == '\n') {
            Image *img = parse_line(line, img_size);
            memset(line, 0, LINE_MAX_SIZE);
            if (!add_to_dataset(img, &dataset, &dataset_max_size, &dataset_cur_size)) {
                free_image(img);
                free_dataset(dataset, dataset_cur_size);
            }
            read_chars = 0;
            if (dataset_cur_size % 1000 == 0) {
                putchar(13);
                printf("Loaded images: %d", dataset_cur_size);
                fflush(stdout);
            }
        } else {
            line[read_chars] = cur_char;
            read_chars++;
        }
    }
    printf(" - OK\n");

    if (read_chars != 0) {
        Image *img = parse_line(line, img_size);
        if (!add_to_dataset(img, &dataset, &dataset_max_size, &dataset_cur_size)) {
            free_image(img);
            free_dataset(dataset, dataset_cur_size);
        }
    }

    free(line);
    fclose(fd);
    *dataset_size = dataset_cur_size;
    
    return dataset;
}

void parse_labels(char *path, Image **dataset, int dataset_size) {
    FILE *fd = fopen(path, "r");
    int cur_char;
    int read = 0;
    while ((cur_char = fgetc(fd)) != EOF) {
        if (cur_char != '\n') {
            if (read == dataset_size) {
                printf("More labels than dataset size!\n");
            }
            dataset[read]->label = cur_char - 48;
            read++;
        }
    }
    fclose(fd);
}

void shallow_copy_dataset(Image **src, Image **dest, int dataset_size) {
    for (int i = 0; i < dataset_size; i++) {
        dest[i] = src[i];
    }
}

void shuffle_dataset(Image **dataset, int dataset_size) {
    for (int i = dataset_size - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        Image *temp = dataset[i];
        dataset[i] = dataset[j];
        dataset[j] = temp;
    }
}