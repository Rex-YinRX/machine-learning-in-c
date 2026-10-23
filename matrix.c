#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m = {0};
    if (rows == 0 || cols == 0) {
        return m;
    }
    m.es = malloc(sizeof(*m.es)*rows*cols);
    if (m.es == NULL){
        return m;
    }

    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    mat_init(m, 0.0f);
    return m;
}

void mat_free(Mat *m)
{
    free(m->es);
    m->es = NULL;
    m->rows = 0;
    m->cols = 0;
    m->stride = 0;
}

void mat_save(FILE *out, Mat m)
{
    const char *magic = "CMAT";
    fwrite(magic, strlen(magic), 1, out);
    fwrite(&m.rows, sizeof(m.rows), 1, out);
    fwrite(&m.cols, sizeof(m.cols), 1, out);
    for (size_t i = 0; i < m.rows; ++i) {
        size_t n = fwrite(&MAT_AT(m, i, 0), sizeof(*m.es), m.cols, out);
        if (n != m.cols) {
            return;
        }
    }
}

Mat mat_load(FILE *in)
{
    Mat m = {0};
    unsigned char magic[4];
    size_t rows = 0;
    size_t cols = 0;

    if (in == NULL) {
        return m;
    }

    fread(magic, 1, sizeof magic, in);
    if (memcmp(magic, "CMAT", sizeof magic) != 0) {
        return m;
    }

    fread(&rows, sizeof rows, 1, in);
    fread(&cols, sizeof cols, 1, in);
    m = mat_alloc(rows, cols);

    for (size_t i = 0; i < m.rows; ++i) {
        size_t n = fread(&MAT_AT(m, i, 0), sizeof *m.es, m.cols, in);

        if (n != m.cols) {
            mat_free(&m);
            return m;
        }
    }

    return m;
}

void mat_init(Mat m, float x)
{
    for(size_t i = 0; i < m.rows; i++){
        for(size_t j = 0; j < m.cols; j++){
            MAT_AT(m, i, j) = x;
        }
    }
}

float rand_float(void)
{
    return (float) rand() / (float) RAND_MAX;
}


void mat_rand(Mat m, float low, float high)
{
    for(size_t i = 0; i < m.rows; ++i){
        for(size_t j = 0; j < m.cols; ++j){
            MAT_AT(m, i, j) = rand_float()*(high - low) + low;
        }
    }
}

int mat_copy(Mat dst, Mat src)
{
    if(dst.rows != src.rows || dst.cols != src.cols){
        return 0;
    }

    for(size_t i = 0; i < dst.rows; ++i){
        for(size_t j = 0; j < dst.cols; ++j){
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
    return 1;
}

int mat_transpose(Mat dst, Mat src)
{
    if(src.es == NULL || dst.es == NULL){
        return 0;
    }
    if(dst.rows != src.cols || dst.cols != src.rows){
        return 0;
    }

    for(size_t i = 0; i < src.rows; ++i){
        for(size_t j = 0; j < src.cols; ++j){
            MAT_AT(dst, j, i) = MAT_AT(src, i, j);
        }
    }
    return 1;
}

void mat_print(Mat m, const char *name)
{
    printf("%s = [\n", name);
    for(size_t i = 0; i < m.rows; i++){
        printf("\t");
        for(size_t j = 0; j < m.cols; j++){
            printf("%.3f ",MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("]\n");
}

int mat_dot(Mat dst, Mat a, Mat b)
{
    if(a.es == NULL || b.es == NULL || dst.es == NULL){
        return 0;
    }
    if(a.cols != b.rows || dst.rows != a.rows || dst.cols != b.cols){
        return 0;
    }

    for(size_t i = 0; i < dst.rows; i++){
        for(size_t j = 0; j < dst.cols; j++){
            MAT_AT(dst, i, j) = 0; 
            for(size_t k = 0; k < a.cols; k++){
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
    return 1;
}

int mat_add_row(Mat dst, Mat a)
{
    if(a.es == NULL || dst.es == NULL){
        return 0;
    }
    if(dst.cols != a.cols || a.rows != 1){
        return 0;
    }

    for(size_t i = 0; i < dst.rows; i++){
        for(size_t j = 0; j < dst.cols; j++){
            MAT_AT(dst, i, j) += MAT_AT(a, 0, j);
        }
    }
    return 1;
}

int mat_sgd_step(Mat param, Mat grad, float learning_rate)
{
    if(param.es == NULL || grad.es == NULL){
        return 0;
    }
    if(param.rows != grad.rows || param.cols != grad.cols){
        return 0;
    }

    for(size_t i = 0; i < param.rows; ++i){
        for(size_t j = 0; j < param.cols; ++j){
            MAT_AT(param, i, j) -= learning_rate * MAT_AT(grad, i, j);
        }
    }
    return 1;
}

int mat_sigmoid(Mat m)
{
    if(m.es == NULL){
        return 0;
    }
    for(size_t i = 0; i < m.rows; i++){
        for(size_t j = 0; j < m.cols; j++){
            MAT_AT(m, i, j) = 1.0f / (1.0f + expf(-MAT_AT(m, i, j)));
        }
    }
    return 1;
}

int mat_relu(Mat m)
{
    if (m.es == NULL) {
        return 0;
    }

    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            float value = MAT_AT(m, i, j);
            MAT_AT(m, i, j) = value > 0.0f ? value : 0.0f;
        }
    }
    return 1;
}

int mat_softmax(Mat m)
{
    if(m.es == NULL){
        return 0;
    }

    for (size_t i = 0; i < m.rows; i++) {
        // 找出当前行的最大值
        float max = MAT_AT(m, i, 0);
        for (size_t j = 1; j < m.cols; j++) {
            if (MAT_AT(m, i, j) > max) {
                max = MAT_AT(m, i, j);
            }
        }

        // 计算指数，写回矩阵并累加
        float sum = 0.0f;
        for (size_t j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = expf(MAT_AT(m, i, j) - max);
            sum += MAT_AT(m, i, j);
        }

        // 归一化为概率
        for (size_t j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) /= sum;
        }
    }

    return 1;
}

size_t mat_argmax_row(Mat m, size_t row)
{
    size_t max_index = 0;
    float max = MAT_AT(m, row, 0);

    for(size_t j = 1; j < m.cols; ++j){
        if(MAT_AT(m, row, j) > max){
            max = MAT_AT(m, row, j);
            max_index = j;
        }
    }
    return max_index;
}
