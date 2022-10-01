//
// Created by mmghf on 9/30/2022.
//

#ifndef C_DATA_LOADER_H
#define C_DATA_LOADER_H


#include <stdlib.h>

#define magic_number_size 4


int number_of_train_images;
int number_of_test_images;
int rows;
int columns;


int readInteger(FILE *fp) {

    unsigned char buffer[4];
    fread(buffer, 1, 4, fp);

    int result = 0;
    for (int i = 0; i < 4; i++) {
        result *= 256;
        result += buffer[i];
    }

    return result;
}


double **readImages(char *address, int t_s) {

    FILE *fp = fopen(address, "rb");
    fseek(fp, magic_number_size, SEEK_SET);

    int images = readInteger(fp);
    rows = readInteger(fp);
    columns = readInteger(fp);
    int pixels = rows * columns;

    int **data = (int **) malloc(images * sizeof(int *));
    for (int i = 0; i < images; i++)
        data[i] = (int *) malloc(pixels * sizeof(int));

    unsigned char pixel;
    for (int i = 0; i < images; i++) {

        for (int j = 0; j < pixels; j++) {

            fread(&pixel, 1, 1, fp);
            data[i][j] = pixel;
        }
    }

    // variable t_s indicates whether this function is reading train data or test data
    if (t_s == 1)
        number_of_train_images = images;
    else
        number_of_test_images = images;

    double **result = (double **) malloc(images * sizeof(double *));
#pragma omp parallel for
    for (int i = 0; i < images; i++) {
        result[i] = (double *) malloc(pixels * sizeof(double));

#pragma omp parallel for
        for (int j = 0; j < pixels; j++)
            result[i][j] = ((double) data[i][j]) / 255;
    }

    return result;
}


double **readLabels(char *address) {

    FILE *fp = fopen(address, "rb");
    fseek(fp, magic_number_size, SEEK_SET);

    int items = readInteger(fp);
    int *labels = (int *) malloc(items * sizeof(int));
    unsigned char label;
    for (int i = 0; i < items; i++) {
        fread(&label, 1, 1, fp);
        labels[i] = label;
    }

    double **result = (double **) malloc(items * sizeof(double *));
#pragma omp parallel for
    for (int i = 0; i < items; i++) {
        result[i] = (double *) malloc(10 * sizeof(double));
        result[i][labels[i] - 1] = 1.0;
    }

    return result;
}


Dataset getDataset() {

    Dataset dataset;
    dataset.train_images = readImages("..\\Datasets\\train-images.idx3-ubyte", 1);
    dataset.train_labels = readLabels("..\\Datasets\\train-labels.idx1-ubyte");
    dataset.test_images = readImages("..\\Datasets\\t10k-images.idx3-ubyte", 0);
    dataset.test_labels = readLabels("..\\Datasets\\t10k-labels.idx1-ubyte");
    dataset.train_samples = number_of_train_images;
    dataset.test_samples = number_of_test_images;
    return dataset;
}

#endif //C_DATA_LOADER_H
