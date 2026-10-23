#ifndef MATRIX_H_
#define MATRIX_H_

#include <stddef.h>
#include <assert.h>
#include <stdio.h>

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Mat;

#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]

Mat mat_alloc(size_t rows, size_t cols);            //创建 rows 行、cols 列的矩阵
void mat_free(Mat *m);                              //释放矩阵拥有的数据，并将该矩阵恢复为空状态
void mat_save(FILE *out, Mat m);
Mat mat_load(FILE *in);
void mat_init(Mat m, float x);                      //将 mat 所指向的所有元素的值变为 x
void mat_rand(Mat m, float low, float high);
int mat_copy(Mat dst, Mat src);                     //将 src 的内容复制给 dst
int mat_transpose(Mat dst, Mat src);
void mat_print(Mat m, const char *name);            //打印 mat 中的内容
#define MAT_PRINT(m) mat_print(m, #m)
int mat_dot(Mat dst, Mat a, Mat b);                 //将矩阵 a 乘 b 的结果输出到 dst 中
int mat_add_row(Mat dst, Mat a);                    //目标矩阵 dst 加上行向量 a
int mat_sgd_step(Mat param, Mat grad, float learning_rate);

int mat_sigmoid(Mat m);
int mat_relu(Mat m);

int mat_softmax(Mat m);
size_t mat_argmax_row(Mat m, size_t row);

float rand_float(void);

#endif //MATRIX_H_
