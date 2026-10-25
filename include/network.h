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
int network_forward(Mat y, Network *net, Mat x);
int network_backward(Mat dx, Network *net, Mat dout);
int network_sgd_step(Network *net, float learning_rate);
void network_rand(Network net);
void network_save(FILE *out, const Network *net);
Network network_load(FILE *in);


#endif //NETWORK_H_
