#include "network.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

#define BATCH_SIZE 1000
#define LEARNING_RATE 0.1f

static int parse_epochs(const char *text, size_t *epochs)
{
    if (text[0] < '0' || text[0] > '9') return 0;
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno == ERANGE || *end != '\0' || value == 0 ||
        value > (unsigned long long)SIZE_MAX) return 0;
    *epochs = (size_t)value;
    return 1;
}

static int finite_parameters(const Network *net)
{
    for (size_t i = 0; i < net->count; ++i) {
        if (net->layers[i].type != LAYER_AFFINE) continue;
        const AffineLayer *layer = &net->layers[i].data.affine;
        Mat parameters[] = {layer->w, layer->b};
        for (size_t p = 0; p < sizeof parameters / sizeof parameters[0]; ++p) {
            Mat m = parameters[p];
            if (m.es == NULL) return 0;
            for (size_t row = 0; row < m.rows; ++row) {
                for (size_t col = 0; col < m.cols; ++col) {
                    if (!isfinite(MAT_AT(m, row, col))) return 0;
                }
            }
        }
    }
    return 1;
}

/* Derive the output width and reject incompatible input/model shapes. */
static size_t model_output_cols(const Network *net, size_t input_cols)
{
    size_t cols = input_cols;
    size_t affine_count = 0;
    for (size_t i = 0; i < net->count; ++i) {
        const Layer *layer = &net->layers[i];
        if (layer->type == LAYER_AFFINE) {
            const AffineLayer *a = &layer->data.affine;
            if (!a->w.es || !a->b.es || !a->dw.es || !a->db.es ||
                a->b.rows != 1 || a->b.cols != a->w.cols ||
                a->dw.rows != a->w.rows || a->dw.cols != a->w.cols ||
                a->db.rows != 1 || a->db.cols != a->w.cols) return 0;
            ++affine_count;
        }
        cols = layer_output_cols(layer, cols);
        if (cols == 0) return 0;
    }
    return affine_count > 0 ? cols : 0;
}

/* Write completely before replacing the original model at the same path. */
static int save_model(const char *path, const Network *net)
{
    size_t length = strlen(path) + sizeof ".tmp";
    char *temporary_path = malloc(length);
    if (temporary_path == NULL) return 0;
    snprintf(temporary_path, length, "%s.tmp", path);
    FILE *out = fopen(temporary_path, "wbx");
    if (out == NULL) {
        perror(temporary_path);
        free(temporary_path);
        return 0;
    }
    network_save(out, net);
    int ok = !ferror(out);
    if (fclose(out) != 0) ok = 0;
    if (!ok) {
        fprintf(stderr, "Cannot write model: %s\n", temporary_path);
        if (remove(temporary_path) != 0) perror(temporary_path);
        free(temporary_path);
        return 0;
    }
#ifdef _WIN32
    ok = MoveFileExA(temporary_path, path,
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
    if (!ok) {
        fprintf(stderr, "Cannot replace model (Windows error %lu): %s\n",
                (unsigned long)GetLastError(), path);
    }
#else
    ok = rename(temporary_path, path) == 0;
    if (!ok) perror(path);
#endif
    if (!ok) fprintf(stderr, "Trained model kept at: %s\n", temporary_path);
    free(temporary_path);
    return ok;
}

int main(int argc, char *argv[])
{
    size_t epoch_count = 0;
    if (argc != 4 || !parse_epochs(argv[3], &epoch_count)) {
        fprintf(stderr, "Usage: \"%s\" model_path data_path epochs\n"
                        "epochs must be a positive integer.\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *data_path = argv[2];
    int exit_code = 1;
    Network net = {0};
    Mat td = {0};
    Mat x_buffer = {0}, score_buffer = {0};
    Mat dscore_buffer = {0}, dx_buffer = {0};
    SoftmaxWithLossLayer loss_buffer = {0};
    size_t batch_labels[BATCH_SIZE];

    FILE *file = fopen(model_path, "rb");
    if (file == NULL) { perror(model_path); goto cleanup; }
    net = network_load(file);
    fclose(file);
    if (net.count == 0) {
        fprintf(stderr, "Cannot load model: %s\n", model_path);
        goto cleanup;
    }

    file = fopen(data_path, "rb");
    if (file == NULL) { perror(data_path); goto cleanup; }
    td = mat_load(file);
    fclose(file);
    if (td.es == NULL || td.rows == 0 || td.cols < 2) {
        fprintf(stderr, "Dataset must contain features and a final label column.\n");
        goto cleanup;
    }

    size_t input_cols = td.cols - 1;
    size_t class_count = model_output_cols(&net, input_cols);
    if (class_count == 0 || !finite_parameters(&net)) {
        fprintf(stderr, "Dataset features do not match a valid model.\n");
        goto cleanup;
    }
    for (size_t row = 0; row < td.rows; ++row) {
        float label = MAT_AT(td, row, input_cols);
        if (!isfinite(label) || label < 0.0f ||
            (double)label >= (double)class_count || floorf(label) != label) {
            fprintf(stderr, "Invalid label at dataset row %zu.\n", row + 1);
            goto cleanup;
        }
        for (size_t col = 0; col < input_cols; ++col) {
            if (!isfinite(MAT_AT(td, row, col))) {
                fprintf(stderr, "Non-finite feature at dataset row %zu.\n", row + 1);
                goto cleanup;
            }
        }
    }

    /* Borrow features and labels from the interleaved dataset, as in main.c. */
    Mat ti = {td.rows, input_cols, td.stride, td.es};
    Mat to = {td.rows, 1, td.stride, td.es + input_cols};
    size_t capacity = td.rows < BATCH_SIZE ? td.rows : BATCH_SIZE;
    x_buffer = mat_alloc(capacity, input_cols);
    score_buffer = mat_alloc(capacity, class_count);
    dscore_buffer = mat_alloc(capacity, class_count);
    dx_buffer = mat_alloc(capacity, input_cols);
    loss_buffer = softmax_with_loss_alloc(capacity, class_count);
    if (!x_buffer.es || !score_buffer.es || !dscore_buffer.es || !dx_buffer.es ||
        !loss_buffer.p.es || !loss_buffer.labels) {
        fprintf(stderr, "Cannot allocate training buffers.\n");
        goto cleanup;
    }

    /* These views change row counts; only the owning buffers are freed. */
    Mat batch_x = x_buffer, batch_scores = score_buffer;
    Mat batch_dscores = dscore_buffer, batch_dx = dx_buffer;
    SoftmaxWithLossLayer batch_loss = loss_buffer;
    time_t training_start = time(NULL);
    printf("samples=%zu,features=%zu,classes=%zu,batch_size=%zu,epochs=%zu,learning_rate=%.3f\n",
           td.rows, input_cols, class_count, capacity, epoch_count, LEARNING_RATE);
    fflush(stdout);

    for (size_t epoch = 0; epoch < epoch_count; ++epoch) {
        for (size_t start = 0; start < ti.rows; ) {
            size_t rows = ti.rows - start;
            if (rows > capacity) rows = capacity;
            batch_x.rows = rows;
            batch_scores.rows = rows;
            batch_dscores.rows = rows;
            batch_dx.rows = rows;
            batch_loss.p.rows = rows;

            for (size_t j = 0; j < rows; ++j) {
                for (size_t k = 0; k < input_cols; ++k) {
                    MAT_AT(batch_x, j, k) = MAT_AT(ti, start + j, k);
                }
                batch_labels[j] = (size_t)MAT_AT(to, start + j, 0);
            }
            if (!network_forward(batch_scores, &net, batch_x) ||
                !softmax_with_loss_forward(&batch_loss, batch_scores, batch_labels, rows) ||
                !isfinite(batch_loss.loss) ||
                !softmax_with_loss_backward(batch_dscores, &batch_loss, 1.0f) ||
                !network_backward(batch_dx, &net, batch_dscores) ||
                !network_sgd_step(&net, LEARNING_RATE)) {
                fprintf(stderr, "Training failed at epoch %zu, sample %zu.\n", epoch + 1, start + 1);
                goto cleanup;
            }
            start += rows;
        }
        if (epoch == 0 || (epoch + 1) % 10 == 0 || epoch + 1 == epoch_count) {
            if (!network_forward(batch_scores, &net, batch_x) ||
                !softmax_with_loss_forward(&batch_loss, batch_scores, batch_labels, batch_x.rows) ||
                !isfinite(batch_loss.loss)) {
                fprintf(stderr, "Cannot evaluate the final batch.\n");
                goto cleanup;
            }
            printf("epoch=%zu,loss=%.6f,time=%.0fs\n", epoch + 1, batch_loss.loss,
                   difftime(time(NULL), training_start));
            fflush(stdout);
        }
    }
    if (!finite_parameters(&net)) {
        fprintf(stderr, "Training produced non-finite parameters; model not saved.\n");
        goto cleanup;
    }
    printf("final_last_batch_loss=%.6f\n", batch_loss.loss);
    if (!save_model(model_path, &net)) goto cleanup;
    printf("model saved: %s\n", model_path);
    exit_code = 0;

cleanup:
    softmax_with_loss_free(&loss_buffer);
    mat_free(&dx_buffer);
    mat_free(&dscore_buffer);
    mat_free(&score_buffer);
    mat_free(&x_buffer);
    mat_free(&td);
    network_free(&net);
    return exit_code;
}
