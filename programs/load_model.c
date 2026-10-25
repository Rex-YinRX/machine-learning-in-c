#include "network.h"
#include <stdio.h>

#define IMG_WIDTH 28
#define IMG_LENGTH 28
#define LABEL_COUNT 10

int main(void)
{
    FILE *model = fopen(".\\model\\mnist_model.bin", "rb");
    if (model == NULL) {
        perror("open file failed");
        return 1;
    }
    Network net = network_load(model);
    fclose(model);
    if (net.count == 0) {
        fprintf(stderr, "load model failed\n");
        network_free(&net);
        return 1;
    }

    FILE *test_file = fopen(".\\data\\test\\test_data", "rb");
    if (test_file == NULL) {
        perror("open test data failed");
        network_free(&net);
        return 1;
    }
    Mat test_td = mat_load(test_file);
    fclose(test_file);
    if (test_td.es == NULL) {
        fprintf(stderr, "load test data failed\n");
        network_free(&net);
        return 1;
    }

    Mat ti = {
        .rows = test_td.rows,
        .cols = IMG_LENGTH*IMG_WIDTH,
        .stride = test_td.stride,
        .es = test_td.es
    };

    Mat to = {
        .rows = test_td.rows,
        .cols = 1,
        .stride = test_td.stride,
        .es = test_td.es + IMG_LENGTH*IMG_WIDTH
    };

    Mat scores = mat_alloc(ti.rows, LABEL_COUNT);
    network_forward(scores, &net, ti);

    size_t correct = 0;
    for (size_t row = 0; row < ti.rows; row++) {
        size_t prediction = mat_argmax_row(scores, row);
        size_t label = (size_t)MAT_AT(to, row, 0);
        if (prediction == label) {
            correct++;
        }
    }

    double test_accuracy = (double)correct / (double)test_td.rows;
    printf("test_accuracy=%.2f%% (%zu/%zu)\n",
           100.0 * test_accuracy, correct, test_td.rows);

    mat_free(&scores);
    mat_free(&test_td);
    network_free(&net);
    return 0;
}
