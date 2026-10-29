#ifndef NETWORK_H_
#define NETWORK_H_

#include "layer.h"



#define DA_INIT_CAP 128

#define da_append(da, item)                                                             \
    do{                                                                                 \
        if((da)->count >= (da)->capacity){                                              \
            (da)->capacity = (da)->capacity == 0 ? DA_INIT_CAP : (da)->capacity*2;      \
            (da)->layers = realloc((da)->layers, (da)->capacity*sizeof(*(da)->layers)); \
            assert((da)->layers != NULL && "Buy more RAM lol");                         \
        }                                                                               \
                                                                                        \
        (da)->layers[(da)->count++] = (item);                                           \
    }while(0)                                                                           \

/*
如何添加一个新的 layer
1. 在 layer.h 添加结构体、函数声明，并扩展 LayerType 和 Layer 联合体。新类型放在枚举末尾
2. 在 layer.c 实现该层的创建、销毁、前向和反向
3. 在统一接口层 Layer 补齐创建对应层函数，以及销毁、前向、反向、参数更新、输入输出形状的函数
4. 在 network.h/c 中声明并实现 network_add_xxx，在 network_struct 中增加对应分支
5. 在network_save 和 network_load 中保存类型和必要参数，加载时重建层。缓存不保存。有参数的层还需处理初始化
*/
    
typedef struct {
    Layer *layers;
    size_t count;
    size_t capacity;
} Network;

Network network_alloc(size_t capacity);

Network network_struct(const int arch[], size_t arch_count, const LayerType layers[], size_t layer_count);
void network_free(Network *net);
int network_add_affine(Network *net, size_t input_dim, size_t output_dim);
int network_add_sigmoid(Network *net);
int network_add_relu(Network *net);
int network_forward(Mat y, Network *net, Mat x);
int network_backward(Mat dx, Network *net, Mat dout);
int network_sgd_step(Network *net, float learning_rate);
void network_rand(Network net);
void network_save(FILE *out, const Network *net);
Network network_load(FILE *in);


#endif //NETWORK_H_
