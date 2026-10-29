#ifndef LAYER_H_
#define LAYER_H_

#include "matrix.h"

/* 全连接层：计算 XW + b，拥有参数、梯度和输入缓存 */
typedef struct {
    Mat w;              //权重矩阵，形状为 input_dim × output_dim
    Mat b;              //偏置矩阵，形状为 1 × output_dim
    Mat dw;             //权重梯度，与 w 形状相同
    Mat db;             //偏置梯度，与 b 形状相同
    Mat x_cache;        // 层拥有的前向输入缓存，首次前向时分配
    int cache_valid;    // 最近一次前向计算成功后为 1
} AffineLayer;

/*全连接层操作函数*/
AffineLayer affine_alloc(size_t input_dim, size_t output_dim);          //建全连接层，分配参数和梯度矩阵
void affine_free(AffineLayer *layer);                                   //释放参数、梯度和输入缓存
int affine_forward(Mat dst, AffineLayer *layer, Mat input);             //计算 input*w + b 写入 dst 并将输入缓存
int affine_backward_input(Mat dst, const AffineLayer *layer, Mat dout); //计算输入梯度 dout × w_T，写入 dst
int affine_backward_weight(Mat dst, Mat input, Mat dout);               //计算权重梯度 input_T × dout，写入 dst
int affine_backward_bias(Mat dst, Mat dout);                            //将 dout 的各行相加，得到偏置梯度并写入 dst
int affine_backward(Mat dx, AffineLayer *layer, Mat dout);              //计算输入梯度 dx，并将参数梯度写入层的 dw、db
int affine_sgd_step(AffineLayer *layer, float learning_rate);           //根据梯度和学习率更新权重与偏置

/* Sigmoid 激活层：保存前向输出，供反向传播计算导数 */
typedef struct {
    Mat output_cache;   //层拥有的前向输出缓存
    int cache_valid;    //最近一次前向计算成功后为 1
} SigmoidLayer;

/*Sigmoid 激活层操作函数*/
SigmoidLayer sigmoid_alloc(void);                                   //初始化 Sigmoid 层，缓存初始为空
void sigmoid_free(SigmoidLayer *layer);                             //释放输出缓存并清空缓存标记
int sigmoid_forward(Mat dst, SigmoidLayer *layer, Mat input);       //计算 Sigmoid，写入 dst 并缓存输出
int sigmoid_backward(Mat dx, const SigmoidLayer *layer, Mat dout);  //利用缓存输出计算输入梯度，写入 dx

typedef struct {
    Mat output_cache;   //层拥有的前向输出缓存
    int cache_valid;    //最近一次前向计算成功后为 1
} ReluLayer;

ReluLayer relu_alloc(void);
void relu_free(ReluLayer *layer);
int relu_forward(Mat dst, ReluLayer *layer,Mat input);
int relu_backward(Mat dst, const ReluLayer *layer, Mat dout);


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



typedef enum {
    LAYER_AFFINE,
    LAYER_SIGMOID,
    LAYER_RELU
} LayerType;

typedef struct {
    LayerType type;

    union {
        AffineLayer affine;
        SigmoidLayer sigmoid;
        ReluLayer relu;
    } data;
} Layer;

Layer layer_affine_alloc(size_t input_dim, size_t output_dim);
Layer layer_sigmoid_alloc(void);
Layer layer_relu_alloc(void);
void layer_free(Layer *layer);
int layer_forward(Mat dst, Layer *layer, Mat input);
int layer_backward(Mat dx, Layer *layer, Mat dout);
int layer_sgd_step(Layer *layer, float learning_rate);
size_t layer_output_cols(const Layer *layer, size_t input_cols);
size_t layer_input_cols(const Layer *layer, size_t dout_cols);

#endif //LAYER_H_
