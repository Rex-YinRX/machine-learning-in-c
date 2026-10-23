#ifndef LAYER_H_
#define LAYER_H_

#include "matrix.h"

typedef struct {
    Mat w; 
    Mat b;  
    Mat dw;
    Mat db;
    Mat x_cache;       /* 层拥有的前向输入副本，首次前向时分配。 */
    int cache_valid;   /* 最近一次前向计算成功后为 1。 */
} AffineLayer;

AffineLayer affine_alloc(size_t input_dim, size_t output_dim);
void affine_free(AffineLayer *layer);
int affine_forward(Mat dst, AffineLayer *layer, Mat input);
int affine_backward_input(Mat dst, const AffineLayer *layer, Mat dout);
int affine_backward_weight(Mat dst, Mat input, Mat dout);
int affine_backward_bias(Mat dst, Mat dout);
int affine_backward(Mat dx, AffineLayer *layer, Mat dout);
int affine_sgd_step(AffineLayer *layer, float learning_rate);

typedef struct {
    Mat output_cache;
    int cache_valid;
} SigmoidLayer;

SigmoidLayer sigmoid_alloc(void);
void sigmoid_free(SigmoidLayer *layer);
int sigmoid_forward(Mat dst, SigmoidLayer *layer, Mat input);
int sigmoid_backward(Mat dx, const SigmoidLayer *layer, Mat dout);

typedef struct {
    Mat p;
    size_t *labels;
    float loss;
    int cache_valid;
} SoftmaxWithLossLayer;

SoftmaxWithLossLayer softmax_with_loss_alloc(size_t batch_size, size_t class_count);
void softmax_with_loss_free(SoftmaxWithLossLayer *layer);
int softmax_with_loss_forward(SoftmaxWithLossLayer *layer, Mat scores, const size_t *labels, size_t count);
int softmax_with_loss_backward(Mat dst, const SoftmaxWithLossLayer *layer, float dout);

int cross_entropy_loss(float *loss, Mat p,const size_t *labels, size_t count);

/*
int relu_forward(Mat dst, Mat input);
int relu_backward(Mat dst, Mat output, Mat dout);
*/


typedef enum {
    LAYER_AFFINE,
    LAYER_SIGMOID
} LayerType;

typedef struct {
    LayerType type;

    union {
        AffineLayer affine;
        SigmoidLayer sigmoid;
    } data;
} Layer;

Layer layer_affine_alloc(size_t input_dim, size_t output_dim);
Layer layer_sigmoid_alloc(void);
void layer_free(Layer *layer);
int layer_forward(Mat dst, Layer *layer, Mat input);
int layer_backward(Mat dx, Layer *layer, Mat dout);
int layer_sgd_step(Layer *layer, float learning_rate);
size_t layer_output_cols(const Layer *layer, size_t input_cols);
size_t layer_input_cols(const Layer *layer, size_t dout_cols);

#endif //LAYER_H_
