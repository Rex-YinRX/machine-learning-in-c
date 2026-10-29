#include "layer.h"
#include <math.h>
#include <stdlib.h>

AffineLayer affine_alloc(size_t input_dim, size_t output_dim)
{
    AffineLayer layer = {0};

    layer.w = mat_alloc(input_dim, output_dim);
    if (layer.w.es == NULL) {
        return layer;
    }
    layer.b = mat_alloc(1, output_dim);
    if (layer.b.es == NULL) {
        affine_free(&layer);
        return layer;
    }
    layer.dw = mat_alloc(input_dim, output_dim);
    if (layer.dw.es == NULL) {
        affine_free(&layer);
        return layer;
    }
    layer.db = mat_alloc(1, output_dim);
    if (layer.db.es == NULL) {
        affine_free(&layer);
        return layer;
    }
    return layer;
}

void affine_free(AffineLayer *layer)
{
    mat_free(&layer->w);
    mat_free(&layer->b);
    mat_free(&layer->dw);
    mat_free(&layer->db);
    mat_free(&layer->x_cache);
    layer->cache_valid = 0;
}

int affine_forward(Mat dst, AffineLayer *layer, Mat input)
{
    if (layer == NULL) {
        return 0;
    }

    layer->cache_valid = 0;
    /* input has independent storage, so it survives releasing the old cache. */
    mat_free(&layer->x_cache);
    layer->x_cache = mat_alloc(input.rows, input.cols);
    if (layer->x_cache.es == NULL) {
        return 0;
    }

    int ok = mat_copy(layer->x_cache, input);
    if (ok) {
        ok = mat_dot(dst, input, layer->w);
    }
    if (ok) {
        ok = mat_add_row(dst, layer->b);
    }
    layer->cache_valid = ok;
    return ok;
}

int affine_backward_input(Mat dst, const AffineLayer *layer, Mat dout)
{
    if (layer == NULL) {
        return 0;
    }

    Mat w_t = mat_alloc(layer->w.cols, layer->w.rows);
    if (w_t.es == NULL) {
        return 0;
    }

    int ok = mat_transpose(w_t, layer->w);
    if (ok) {
        ok = mat_dot(dst, dout, w_t);
    }
    mat_free(&w_t);
    return ok;
}

int affine_backward_weight(Mat dst, Mat input, Mat dout)
{
    Mat input_t = mat_alloc(input.cols, input.rows);
    if (input_t.es == NULL) {
        return 0;
    }

    int ok = mat_transpose(input_t, input);
    if (ok) {
        ok = mat_dot(dst, input_t, dout);
    }
    mat_free(&input_t);
    return ok;
}

int affine_backward_bias(Mat dst, Mat dout)
{
    if (dst.es == NULL || dout.es == NULL) {
        return 0;
    }
    if (dst.rows != 1 || dst.cols != dout.cols) {
        return 0;
    }

    for (size_t j = 0; j < dout.cols; ++j) {
        float sum = 0.0f;
        for (size_t i = 0; i < dout.rows; ++i) {
            sum += MAT_AT(dout, i, j);
        }
        MAT_AT(dst, 0, j) = sum;
    }
    return 1;
}

int affine_backward(Mat dx, AffineLayer *layer, Mat dout)
{
    int ok = affine_backward_bias(layer->db, dout);
    if (ok) {
        ok = affine_backward_weight(layer->dw, layer->x_cache, dout);
    }
    if (ok) {
        ok = affine_backward_input(dx, layer, dout);
    }
    return ok;
}

int affine_sgd_step(AffineLayer *layer, float learning_rate)
{
    mat_sgd_step(layer->w, layer->dw, learning_rate);
    mat_sgd_step(layer->b, layer->db, learning_rate);
    return 1;
}

SigmoidLayer sigmoid_alloc(void){
    SigmoidLayer layer = {0};
    return layer;
}

void sigmoid_free(SigmoidLayer *layer)
{
    mat_free(&layer->output_cache);   
    layer->cache_valid = 0;
}

int sigmoid_forward(Mat dst, SigmoidLayer *layer, Mat input)
{
    if (layer == NULL) {
        return 0;
    }

    layer->cache_valid = 0;
    if (dst.es == NULL || input.es == NULL ||
        input.rows == 0 || input.cols == 0 ||
        dst.rows != input.rows || dst.cols != input.cols) {
        return 0;
    }

    mat_free(&layer->output_cache);
    layer->output_cache = mat_alloc(input.rows, input.cols);
    if (layer->output_cache.es == NULL) {
        return 0;
    }

    int ok = mat_copy(dst, input);
    if (ok) {
        ok = mat_sigmoid(dst);
    }
    if (ok) {
        ok = mat_copy(layer->output_cache, dst);
    }
    layer->cache_valid = ok;
    return ok;
}

int sigmoid_backward(Mat dx, const SigmoidLayer *layer, Mat dout)
{
    if (layer == NULL || !layer->cache_valid) {
        return 0;
    }

    Mat output = layer->output_cache;
    for (size_t i = 0; i < output.rows; ++i) {
        for (size_t j = 0; j < output.cols; ++j) {
            float a = MAT_AT(output, i, j);
            MAT_AT(dx, i, j) = MAT_AT(dout, i, j) * a * (1.0f - a);
        }
    }
    return 1;
}

ReluLayer relu_alloc(void)
{
    ReluLayer layer = {0};
    return layer;
}

void relu_free(ReluLayer *layer)
{
    mat_free(&layer->output_cache);   
    layer->cache_valid = 0;
}

int relu_forward(Mat dst, ReluLayer *layer, Mat input)
{
    if (layer == NULL) {
        return 0;
    }

    layer->cache_valid = 0;
    if (dst.es == NULL || input.es == NULL) {
        return 0;
    }
    if (dst.rows != input.rows || dst.cols != input.cols) {
        return 0;
    }

    mat_free(&layer->output_cache);
    layer->output_cache = mat_alloc(input.rows, input.cols);
    if (layer->output_cache.es == NULL) {
        return 0;
    }

    int ok = mat_copy(dst, input);
    if (ok) {
        ok = mat_relu(dst);
    }
    if (ok) {
        ok = mat_copy(layer->output_cache, dst);
    }
    layer->cache_valid = ok;

    return ok;
}

int relu_backward(Mat dst, const ReluLayer *layer, Mat dout)
{
    if (layer == NULL || !layer->cache_valid) {
        return 0;
    }
    if (dst.es == NULL || layer->output_cache.es == NULL || dout.es == NULL) {
        return 0;
    }
    if (dst.rows != layer->output_cache.rows || dst.cols != layer->output_cache.cols ||
        dout.rows != layer->output_cache.rows || dout.cols != layer->output_cache.cols) {
        return 0;
    }

    for (size_t i = 0; i < layer->output_cache.rows; ++i) {
        for (size_t j = 0; j < layer->output_cache.cols; ++j) {
            MAT_AT(dst, i, j) = MAT_AT(layer->output_cache, i, j) > 0.0f
                              ? MAT_AT(dout, i, j) : 0.0f;
        }
    }
    return 1;
}

SoftmaxWithLossLayer softmax_with_loss_alloc(size_t batch_size, size_t class_count)
{
    SoftmaxWithLossLayer layer = {0};

    layer.p = mat_alloc(batch_size, class_count);
    if (layer.p.es == NULL) {
        return layer;
    }
    layer.labels = malloc(sizeof(*layer.labels) * batch_size);
    if (layer.labels == NULL) {
        mat_free(&layer.p);
        return layer;
    }
    return layer;
}

void softmax_with_loss_free(SoftmaxWithLossLayer *layer)
{
    mat_free(&layer->p);
    free(layer->labels);
    layer->labels = NULL;
    layer->loss = 0.0f;
    layer->cache_valid = 0;
}

int softmax_with_loss_forward(SoftmaxWithLossLayer *layer, Mat scores, const size_t *labels, size_t count)
{
    mat_copy(layer->p, scores);
    mat_softmax(layer->p);
    cross_entropy_loss(&layer->loss, layer->p, labels, count);
    
    for (size_t i = 0; i < count; ++i) {
        layer->labels[i] = labels[i];
    }
    layer->cache_valid = 1;
    return 1;
}

int softmax_with_loss_backward(Mat dst, const SoftmaxWithLossLayer *layer, float dout)
{
    float scale = dout / (float)layer->p.rows;

    for (size_t i = 0; i < layer->p.rows; ++i) {
        for (size_t j = 0; j < layer->p.cols; ++j) {
            float grad = MAT_AT(layer->p, i, j);
            if (j == layer->labels[i]) {
                grad -= 1.0f;
            }
            MAT_AT(dst, i, j) = grad * scale;
        }
    }
    return 1;
}

int cross_entropy_loss(float *loss, Mat p, const size_t *labels, size_t count)
{
    if (loss == NULL || p.es == NULL || labels == NULL) {
        return 0;
    }
    if (count == 0 || count != p.rows || p.cols == 0) {
        return 0;
    }

    float total = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        if (labels[i] >= p.cols) {
            return 0;
        }

        //计算平均交叉熵，正确类别概率的下限为 1e-7f。
        float prob = MAT_AT(p, i, labels[i]);
        if (prob < 1e-7f) {
            prob = 1e-7f;
        }
        total -= logf(prob);
    }

    *loss = total / (float)count;
    return 1;
}

Layer layer_affine_alloc(size_t input_dim, size_t output_dim)
{
    Layer layer = {0};
    layer.type = LAYER_AFFINE;
    layer.data.affine = affine_alloc(input_dim, output_dim);
    return layer;
}

Layer layer_sigmoid_alloc(void)
{
    Layer layer = {0};
    layer.type = LAYER_SIGMOID;
    layer.data.sigmoid = sigmoid_alloc();
    return layer;
}

Layer layer_relu_alloc(void)
{
    Layer layer = {0};
    layer.type = LAYER_RELU;
    layer.data.relu = relu_alloc();
    return layer;
}

void layer_free(Layer *layer)
{
    if (layer == NULL) {
        return;
    }

    switch (layer->type) {
        case LAYER_AFFINE:
            affine_free(&layer->data.affine);
            break;
        case LAYER_SIGMOID:
            sigmoid_free(&layer->data.sigmoid);
            break;
        case LAYER_RELU:
            relu_free(&layer->data.relu);
            break;
    }
}

int layer_forward(Mat dst, Layer *layer, Mat input)
{
    if (layer == NULL) {
        return 0;
    }

    switch (layer->type) {
        case LAYER_AFFINE:
            return affine_forward(dst, &layer->data.affine, input);
        case LAYER_SIGMOID:
            return sigmoid_forward(dst, &layer->data.sigmoid, input);
        case LAYER_RELU:
            return relu_forward(dst, &layer->data.relu, input);
        default:
            return 0;
    }
}

int layer_backward(Mat dx, Layer *layer, Mat dout)
{
    if(layer == NULL){
        return 0;
    }
    switch (layer->type) {
        case LAYER_AFFINE:
            return affine_backward(dx, &layer->data.affine, dout);
        case LAYER_SIGMOID:
            return sigmoid_backward(dx, &layer->data.sigmoid, dout);
        case LAYER_RELU:
            return relu_backward(dx, &layer->data.relu, dout);
        default:
            return 0;
    }
}

int layer_sgd_step(Layer *layer, float learning_rate)
{
    if(layer == NULL){
        return 0;
    }
    switch (layer->type) {
        case LAYER_AFFINE:
            return affine_sgd_step(&layer->data.affine, learning_rate);
        case LAYER_SIGMOID:
            return 1;
        case LAYER_RELU:
            return 1;
        default:
            return 0;
    }
}

size_t layer_output_cols(const Layer *layer, size_t input_cols)
{
    if (layer == NULL) {
        return 0;
    }

    switch (layer->type) {
        case LAYER_AFFINE:
            if (input_cols != layer->data.affine.w.rows) {
                return 0;
            }
            return layer->data.affine.w.cols;
        case LAYER_SIGMOID:
            return input_cols;
        case LAYER_RELU:
            return input_cols;
        default:
            return 0;
    }
}

size_t layer_input_cols(const Layer *layer, size_t dout_cols)
{
    if (layer == NULL || dout_cols == 0) {
        return 0;
    }

    switch (layer->type) {
        case LAYER_AFFINE:
            if (dout_cols != layer->data.affine.w.cols) {
                return 0;
            }
            return layer->data.affine.w.rows;
        case LAYER_SIGMOID:
            return dout_cols;
        case LAYER_RELU:
            return dout_cols;
        default:
            return 0;
    }
}