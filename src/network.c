#include "network.h"
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

Network network_alloc(size_t capacity)
{
    Network net = {0};
    net.layers = malloc(capacity * sizeof(*net.layers));
    net.capacity = capacity;
    return net;
}

Network network_struct(const int arch[], size_t arch_count, const LayerType layers[], size_t layer_count)
{
    Network net = {0};

    size_t affine_count = 0;

    for (size_t j = 0; j < layer_count; ++j) {
        if (layers[j] == LAYER_AFFINE) {
            affine_count++;
        }
    }

    if (arch_count != affine_count + 1) {
        return net;
    }

    net = network_alloc(layer_count);
    size_t i = 0;
    for (size_t j = 0; j < layer_count; j++) {
        switch (layers[j]) {
        case LAYER_AFFINE:
            network_add_affine(&net, (size_t)arch[i], (size_t)arch[i + 1]);
            i++;
            break;
        case LAYER_SIGMOID:
            network_add_sigmoid(&net);
            break;
        case LAYER_RELU:
            network_add_relu(&net);
            break;
        }
    }

    return net;
}

void network_free(Network *net)
{
    for(size_t i = 0; i < net->count; i++){
        layer_free(net->layers + i);
    }
    free(net->layers);
    net->layers = NULL;
    net->count = 0;
    net->capacity = 0;
}

int network_add_affine(Network *net, size_t input_dim, size_t output_dim)
{
    da_append(net, layer_affine_alloc(input_dim, output_dim));
    return 1;
}

int network_add_sigmoid(Network *net)
{
    da_append(net, layer_sigmoid_alloc());
    return 1;
}

int network_add_relu(Network *net)
{
    da_append(net, layer_relu_alloc());
    return 1;
}

int network_forward(Mat y, Network *net, Mat x)
{
    if (net == NULL || net->count == 0) {
        return 0;
    }

    Mat current = x;
    int current_owned = 0;

    for(size_t i = 0; i < net->count; i++){
        Layer *layer = &net->layers[i];
        size_t out_cols = layer_output_cols(layer, current.cols);

        Mat next = mat_alloc(current.rows, out_cols);

        layer_forward(next, layer, current);

        if(current_owned == 1){
            mat_free(&current);
        }
        current = next;
        current_owned = 1;

    }
    int ok = mat_copy(y, current);
    mat_free(&current);
    return ok;
}

int network_backward(Mat dx, Network *net, Mat dout)
{
    if (net == NULL || net->count == 0) {
        return 0;
    }

    Mat current = dout;
    int current_owned = 0;

    for (size_t i = net->count; i > 0; --i) {
        Layer *layer = &net->layers[i - 1];
        size_t out_cols = layer_input_cols(layer, current.cols);

        Mat next = mat_alloc(current.rows, out_cols);

        layer_backward(next, layer, current);

        if(current_owned == 1){
            mat_free(&current);
        }
        current = next;
        current_owned = 1;

    }
    int ok = mat_copy(dx, current);
    mat_free(&current);
    return ok;
}

int network_sgd_step(Network *net, float learning_rate)
{
    if (net == NULL || net->count == 0) {
        return 0;
    }

    for (size_t i = 0; i < net->count; ++i) {
        if (!layer_sgd_step(&net->layers[i], learning_rate)) {
            return 0;
        }
    }

    return 1;
}

void network_rand(Network net)
{
    for (size_t i = 0; i < net.count; ++i) {
        Layer *layer = &net.layers[i];

        if (layer->type == LAYER_AFFINE) {
            AffineLayer *affine = &layer->data.affine;

            mat_rand(affine->w, -0.1f, 0.1f);
        }
    }
}

void network_save(FILE *out, const Network *net)
{
    const char magic[4] = {'C', 'N', 'E', 'T'};
    uint32_t version = 1;
    uint32_t layer_count = (uint32_t)net->count;

    fwrite(magic, 1, sizeof magic, out);
    fwrite(&version, sizeof version, 1, out);
    fwrite(&layer_count, sizeof layer_count, 1, out);

    for (size_t i = 0; i < net->count; ++i) {
        const Layer *layer = &net->layers[i];

        uint32_t type = (uint32_t)layer->type;
        fwrite(&type, sizeof type, 1, out);

        if (layer->type == LAYER_AFFINE) {
            const AffineLayer *affine = &layer->data.affine;

            mat_save(out, affine->w);
            mat_save(out, affine->b);
        }
    }
}

Network network_load(FILE *in)
{
    Network net = {0};

    unsigned char magic[4];
    uint32_t version = 0;
    uint32_t layer_count = 0;
    
    fread(magic, 1, sizeof magic, in);
    if(memcmp(magic, "CNET", sizeof magic) != 0){
        return net;
    }

    fread(&version, sizeof version, 1, in);
    if(version != 1){
        return net;
    }

    fread(&layer_count, sizeof layer_count, 1, in);
    net = network_alloc(layer_count);

    for (uint32_t i = 0; i < layer_count; i++){
        uint32_t type = 0;
        fread(&type, sizeof type, 1, in);
        if (type == LAYER_SIGMOID) {
            network_add_sigmoid(&net);
            continue;
        }
        if(type == LAYER_RELU){
            network_add_relu(&net);
            continue;
        }
        if (type != LAYER_AFFINE) {
            network_free(&net);
            return net;
        }

        Mat w = mat_load(in);
        Mat b = mat_load(in);

        network_add_affine(&net, w.rows, w.cols);
        AffineLayer *affine = &net.layers[net.count - 1].data.affine;
        mat_copy(affine->w, w);
        mat_copy(affine->b, b);
        mat_free(&w);
        mat_free(&b);

    }

    return net;
}