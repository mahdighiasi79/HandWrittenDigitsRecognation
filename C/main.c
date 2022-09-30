#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "matrix.h"
#include "dataTypes.h"
#include "data_loader.h"

#define learning_rate 0.01
#define batch_size 50
#define number_of_epochs 5


typedef struct ann {
    double **w1;
    double **w2;
    double **w3;
    double **b1;
    double **b2;
    double **b3;
} Model;


void initialize_parameters(Model *model) {

    time_t t = time(NULL);
    srand(t);

    model->w1 = (double **) malloc(16 * sizeof(double *));
#pragma omp parallel for
    for (int i = 0; i < 16; i++) {
        model->w1[i] = (double *) malloc(784 * sizeof(double));
#pragma omp parallel for
        for (int j = 0; j < 784; j++) {
            model->w1[i][j] = normal_distribution((double) rand(), 0, 1);
        }
    }

    model->w2 = (double **) malloc(16 * sizeof(double *));
#pragma omp parallel for
    for (int i = 0; i < 16; i++) {
        model->w2[i] = (double *) malloc(16 * sizeof(double));
#pragma omp parallel for
        for (int j = 0; j < 16; j++) {
            model->w2[i][j] = normal_distribution((double) rand(), 0, 1);
        }
    }

    model->w3 = (double **) malloc(10 * sizeof(double *));
#pragma omp parallel for
    for (int i = 0; i < 10; i++) {
        model->w3[i] = (double *) malloc(16 * sizeof(double));
#pragma omp parallel for
        for (int j = 0; j < 16; j++) {
            model->w3[i][j] = normal_distribution((double) rand(), 0, 1);
        }
    }

    model->b1 = (double **) malloc(16 * sizeof(double *));
    for (int i = 0; i < 16; i++)
        model->b1[0] = (double *) malloc(sizeof(double));

    model->b2 = (double **) malloc(16 * sizeof(double *));
    for (int i = 0; i < 16; i++)
        model->b2[0] = (double *) malloc(sizeof(double));

    model->b3 = (double **) malloc(10 * sizeof(double *));
    for (int i = 0; i < 10; i++)
        model->b3[0] = (double *) malloc(sizeof(double));
}


double **relu(double **matrix, int row, int column) {
    double **result = (double **) malloc(row * sizeof(double *));

#pragma omp parallel for
    for (int i = 0; i < row; i++) {
        result[i] = (double *) malloc(column * sizeof(double));

#pragma omp parallel for
        for (int j = 0; j < column; j++) {
            if (matrix[i][j] < 0)
                result[i][j] = 0;
            else
                result[i][j] = matrix[i][j];
        }
    }
    return result;
}


double **relu_derivative(double **matrix, int row, int column) {
    double **result = (double **) malloc(row * sizeof(double *));

#pragma omp parallel for
    for (int i = 0; i < row; i++) {
        result[i] = (double *) malloc(column * sizeof(double));

#pragma omp parallel for
        for (int j = 0; j < column; j++) {
            if (matrix[i][j] < 0)
                result[i][j] = 0;
            else
                result[i][j] = 1;
        }
    }
    return result;
}


Cache forward_prop(double **inputs, int m, Model model) {
    Cache cache;

    double **z1 = matrix_multiplication(model.w1, inputs, 16, 784, m);
    z1 = matrix_broadcast_addition(z1, model.b1, 16, m);
    double **a1 = relu(z1, 16, m);
    cache.z1 = z1;
    cache.a1 = a1;

    double **z2 = matrix_multiplication(model.w2, a1, 16, 16, m);
    z2 = matrix_broadcast_addition(z2, model.b2, 16, m);
    double **a2 = relu(z2, 16, m);
    cache.z2 = z2;
    cache.a2 = a2;

    double **z3 = matrix_multiplication(model.w3, a2, 10, 16, m);
    z3 = matrix_broadcast_addition(z3, model.b3, 10, m);
    double **a3 = relu(z3, 10, m);
    cache.z3 = z3;
    cache.a3 = a3;

    return cache;
}


int *predict(Model model, double **inputs, int m) {
    Cache cache = forward_prop(inputs, m, model);
    double **y = cache.a3;
    int *predicted_values = arg_max(y, 1, 10, m);
    return predicted_values;
}


double loss(double **inputs, double **labels, int m, Model model) {
    Cache cache = forward_prop(inputs, m, model);
    double **y = cache.a3;
    double **loss = matrix_subtraction(y, labels, 10, m);
    loss = power_matrix(loss, 2, 10, m);
    loss = sum_axis(loss, 0, 10, m);
    loss = sum_axis(loss, 1, 1, m);
    double result = loss[0][0];
    result /= m;
    return result;
}


Grads back_prop(double **inputs, double **labels, int m, Cache cache, Model model) {
    Grads grads;

    double **a3_labels = matrix_subtraction(cache.a3, labels, 10, m);
    double **g_prime_z3 = relu_derivative(cache.z3, 10, m);
    double **a2_t = transpose(cache.a2, 16, m);
    double **da3 = matrix_constant_mul(a3_labels, 2, 10, m);
    double **db3 = element_wise_mul(da3, g_prime_z3, 10, m);
    double **dw3 = matrix_multiplication(db3, a2_t, 10, m , 16);

    double **w3_t = transpose(model.w3, 10 , 16);
    double **g_prime_z3_ElementWiseMul_da3 = element_wise_mul(g_prime_z3, da3, 10, m);
    double **g_prime_z2 = relu_derivative(cache.z2, 16, m);
    double **a1_t = transpose(cache.a1, 16, m);
    double **da2 = matrix_multiplication(w3_t, g_prime_z3_ElementWiseMul_da3, 16, 10, m);
    double **db2 = element_wise_mul(da2, g_prime_z2, 16, m);
    double **dw2 = matrix_multiplication(db2, a1_t, 16, m, 16);

    double **w2_t = transpose(model.w2, 16 , 16);
    double **g_prime_z2_ElementWiseMul_da2 = element_wise_mul(g_prime_z2, da2, 16, m);
    double **g_prime_z1 = relu_derivative(cache.z1, 16, m);
    double **a0_t = transpose(inputs, 784, m);
    double **da1 = matrix_multiplication(w2_t, g_prime_z2_ElementWiseMul_da2, 16, 16, m);
    double **db1 = element_wise_mul(da1, g_prime_z1, 16, m);
    double **dw1 = matrix_multiplication(db1, a0_t, 16, m, 784);

    double constant = 1 / (double) m;
    dw1 = matrix_constant_mul(dw1, constant, 16, 784);
    dw2 = matrix_constant_mul(dw2, constant, 16, 16);
    dw3 = matrix_constant_mul(dw3, constant, 10, 16);

    db1 = sum_axis(db1, 1, 16, m);
    db2 = sum_axis(db2, 1, 16, m);
    db3 = sum_axis(db3, 1, 10, m);
    db1 = matrix_constant_mul(db1, constant, 16, 1);
    db2 = matrix_constant_mul(db2, constant, 16, 1);
    db3 = matrix_constant_mul(db3, constant, 10, 1);

    grads.dw1 = dw1;
    grads.dw2 = dw2;
    grads.dw3 = dw3;
    grads.db1 = db1;
    grads.db2 = db2;
    grads.db3 = db3;

    return grads;
}


void optimizer_step(double **inputs, double **labels, int m, Model *model) {
    Cache cache = forward_prop(inputs, m, *model);
    Grads grads = back_prop(inputs, labels, m, cache, *model);

    double **lr_multiply_dw1 = matrix_constant_mul(grads.dw1, learning_rate, 16, 784);
    double **lr_multiply_dw2 = matrix_constant_mul(grads.dw2, learning_rate, 16, 16);
    double **lr_multiply_dw3 = matrix_constant_mul(grads.dw3, learning_rate, 10, 16);
    double **lr_multiply_db1 = matrix_constant_mul(grads.db1, learning_rate, 16, 1);
    double **lr_multiply_db2 = matrix_constant_mul(grads.db2, learning_rate, 16, 1);
    double **lr_multiply_db3 = matrix_constant_mul(grads.db3, learning_rate, 10, 1);

    model->w1 = matrix_subtraction(model->w1, lr_multiply_dw1, 16, 784);
    model->w2 = matrix_subtraction(model->w2, lr_multiply_dw2, 16, 16);
    model->w3 = matrix_subtraction(model->w3, lr_multiply_dw3, 10, 16);
    model->b1 = matrix_subtraction(model->b1, lr_multiply_db1, 16, 1);
    model->b2 = matrix_subtraction(model->b2, lr_multiply_db2, 16, 1);
    model->b3 = matrix_subtraction(model->b3, lr_multiply_db3, 10, 1);
}


void sgd(Model *model, double **inputs, double **labels, int m) {
    int number_of_batches = floor((double) m / batch_size);

    for (int i = 0; i < number_of_epochs; i++) {

        inputs = shuffle(inputs, m);

        for (int j = 0; j < number_of_batches; j++) {
            double **x = copy_matrix(inputs, j * batch_size, (j + 1) * batch_size);
            double **y = copy_matrix(labels, j * batch_size, (j + 1) * batch_size);
            x = transpose(x, batch_size, 784);
            y = transpose(y, batch_size, 10);
            optimizer_step(x, y, batch_size, model);
            printf("loss: %f\n", loss(x, y, batch_size, *model));
        }
    }
}


int main() {
    double **train_images = get_train_images();
    double **train_labels = get_train_labels();
    double **test_images = get_test_images();
    double **test_labels = get_test_labels();

    Model *model = (Model *) malloc(sizeof(Model));
    initialize_parameters(model);
    sgd(model, train_images, train_labels, train_samples);

    int *predictions = predict(*model, test_images, test_samples);
    int *true_values = arg_max(test_labels, 0, test_samples, 10);
    int true_answers = 0;

#pragma omp parallel for
    for (int i = 0; i < test_samples; i++) {
        if (predictions[i] == true_values[i])
            true_answers++;
    }

    double m = test_samples;
    double accuracy = ((double) true_answers / m) * 100;
    printf("accuracy: %f", accuracy);
    return 0;
}
