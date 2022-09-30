//
// Created by mmghf on 9/30/2022.
//

#ifndef C_DATA_LOADER_H
#define C_DATA_LOADER_H


#include <stdlib.h>

#define train_samples 60000
#define test_samples 10000


double **get_train_images() {
    double **train_images = (double **) malloc(train_samples * sizeof(double *));
    for (int i = 0; i < train_samples; i++)
        train_images[i] = (double *) malloc(784 * sizeof(double));
    return train_images;
}


double **get_train_labels() {
    double **train_labels = (double **) malloc(train_samples * sizeof(double *));
    for (int i = 0; i < train_samples; i++)
        train_labels[i] = (double *) malloc(10 * sizeof(double));
    return train_labels;
}


double **get_test_images() {
    double **test_images = (double **) malloc(test_samples * sizeof(double *));
    for (int i = 0; i < test_samples; i++)
        test_images[i] = (double *) malloc(784 * sizeof(double));
    return test_images;
}


double **get_test_labels() {
    double **test_labels = (double **) malloc(test_samples * sizeof(double *));
    for (int i = 0; i < test_samples; i++)
        test_labels[i] = (double *) malloc(10 * sizeof(double));
    return test_labels;
}


#endif //C_DATA_LOADER_H
