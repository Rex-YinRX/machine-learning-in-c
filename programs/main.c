#include "network.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define IMG_WIDTH 28
#define IMG_LENGTH 28
#define LABEL_COUNT 10
#define LEARNING_RATE 0.1
#define EPOCH_COUNT 400
#define BATCH_SIZE 1000

double dataset_accuracy(Network *net, Mat data, Mat score_buffer, size_t *correct_count)

{
    size_t correct = 0;
    size_t batch_size = score_buffer.rows;
    size_t input_cols = data.cols - 1;

    for (size_t start = 0; start < data.rows; start += batch_size) {
        size_t rows = data.rows - start;
        if (rows > batch_size) {
            rows = batch_size;
        }

        Mat x = {
            .rows = rows,
            .cols = input_cols,
            .stride = data.stride,
            .es = data.es + start * data.stride
        };

        Mat scores = score_buffer;
        scores.rows = rows;

        network_forward(scores, net, x);

        for (size_t row = 0; row < rows; row++) {
            size_t prediction = mat_argmax_row(scores, row);
            size_t label =
                (size_t)MAT_AT(data, start + row, input_cols);

            if (prediction == label) {
                correct++;
            }
        }
    }


    *correct_count = correct;
    return (double)correct / (double)data.rows;
}

int main(void)
{
    srand(time(0));

    FILE *tdata = fopen(".\\data\\train\\train_data", "rb");
    if (tdata == NULL) {
        perror("open file failed");
        return 1;
    }

    Mat td = mat_load(tdata);
    fclose(tdata);
    Mat ti = {
        .rows = td.rows,
        .cols = IMG_LENGTH*IMG_WIDTH,
        .stride = IMG_LENGTH*IMG_WIDTH + 1,
        .es = td.es
    };

    Mat to = {
        .rows = td.rows,
        .cols = 1,
        .stride = IMG_LENGTH*IMG_WIDTH + 1,
        .es = td.es + IMG_LENGTH*IMG_WIDTH
    };


    const int arch[] = {
        IMG_LENGTH*IMG_WIDTH, 200, 80, LABEL_COUNT
    };
    const LayerType layer_types[] = {
        LAYER_AFFINE, LAYER_SIGMOID, LAYER_AFFINE, LAYER_SIGMOID, LAYER_AFFINE
    };
    Network net = network_struct(arch, sizeof arch / sizeof arch[0], layer_types, sizeof layer_types / sizeof layer_types[0]);
    network_rand(net);

    
    SoftmaxWithLossLayer batch_loss = softmax_with_loss_alloc(BATCH_SIZE, LABEL_COUNT);
    Mat batch_x = mat_alloc(BATCH_SIZE, ti.cols);
    Mat batch_scores = mat_alloc(BATCH_SIZE, LABEL_COUNT);
    Mat batch_dscores = mat_alloc(BATCH_SIZE, LABEL_COUNT);
    Mat batch_dx = mat_alloc(BATCH_SIZE, ti.cols);
    size_t batch_labels[BATCH_SIZE];
    size_t batch_count = ti.rows / BATCH_SIZE;
    

    time_t training_start = time(NULL);
    for (size_t epoch = 0; epoch < EPOCH_COUNT; ++epoch) {
        for(size_t i = 0; i < batch_count; i++){
            for(size_t j = 0; j < BATCH_SIZE; j++){
                for(size_t k = 0; k < IMG_LENGTH*IMG_WIDTH; k++){
                    MAT_AT(batch_x, j, k) = MAT_AT(ti, i*BATCH_SIZE + j, k);
                }
                batch_labels[j] = (size_t)MAT_AT(to, i*BATCH_SIZE + j, 0);
            }
            network_forward(batch_scores, &net, batch_x);
            softmax_with_loss_forward(&batch_loss, batch_scores,
                                     batch_labels, batch_x.rows);


            softmax_with_loss_backward(batch_dscores, &batch_loss, 1.0f);
            network_backward(batch_dx, &net, batch_dscores);
            network_sgd_step(&net, LEARNING_RATE);
        }
        
        if (epoch == 0 || (epoch + 1) % 10 == 0){
            network_forward(batch_scores, &net, batch_x);
            softmax_with_loss_forward(&batch_loss, batch_scores, batch_labels, batch_x.rows);
            time_t end = time(NULL);
            double total_seconds = difftime(end, training_start);
            printf("epoch=%zu,loss=%.6f,time=%.4lfs\n", epoch+1, batch_loss.loss, total_seconds);
        }
        
        
     }


    network_forward(batch_scores, &net, batch_x);
    softmax_with_loss_forward(&batch_loss, batch_scores, batch_labels, batch_x.rows);
    printf("final_last_batch_loss=%.6f\n", batch_loss.loss);

    FILE *test_file = fopen(".\\data\\test\\test_data", "rb");
    if (test_file == NULL) {
        perror("open test data failed");
        return 1;
    }
    Mat test_td = mat_load(test_file);
    fclose(test_file);

    size_t train_correct = 0;
    size_t test_correct = 0;
    double train_accuracy = dataset_accuracy(&net, td, batch_scores, &train_correct);
    double test_accuracy = dataset_accuracy(&net, test_td, batch_scores, &test_correct);
    
    printf("train_accuracy=%.2f%% (%zu/%zu)\n", 100.0 * train_accuracy, train_correct, td.rows);
    printf("test_accuracy=%.2f%% (%zu/%zu)\n", 100.0 * test_accuracy, test_correct, test_td.rows);

    FILE *model_file = fopen(".\\model\\mnist_model.bin", "wb");

    if (model_file == NULL) {
        perror("open model file failed");
    } else {
        network_save(model_file, &net);
        printf("model saved");
        fclose(model_file);
    }

    mat_free(&batch_dx);
    mat_free(&batch_dscores);
    mat_free(&batch_scores);
    mat_free(&batch_x);
    mat_free(&td);
    mat_free(&test_td);
    softmax_with_loss_free(&batch_loss);
    network_free(&net);
    return 0;
}
