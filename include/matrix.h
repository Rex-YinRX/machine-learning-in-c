#ifndef MATRIX_H_
#define MATRIX_H_

#include <stddef.h>
#include <assert.h>
#include <stdio.h>

typedef struct {
    size_t rows;            //矩阵的行数
    size_t cols;            //矩阵的列数
    size_t stride;          //相邻两行起始位置的间隔，以元素个数为单位
    float *es;              //矩阵首个元素的地址，可能指向自有数据或借用的数据
} Mat;

/* 访问矩阵第 i 行、第 j 列的元素，索引从 0 开始，可用于读取或赋值 */
#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]

Mat mat_alloc(size_t rows, size_t cols);            //创建 rows 行、cols 列的矩阵
void mat_free(Mat *m);                              //释放矩阵拥有的数据，并将该矩阵恢复为空状态
void mat_save(FILE *out, Mat m);                    //将矩阵 m 以二进制格式写入文件流 out
Mat mat_load(FILE *in);                             //从二进制文件流 in 读取并创建矩阵
void mat_init(Mat m, float x);                      //将 m 所指向的所有元素的值变为 x
void mat_rand(Mat m, float low, float high);        //将矩阵 m 的所有元素赋为 [low, high] 范围内的随机数
int mat_copy(Mat dst, Mat src);                     //将 src 的内容复制给 dst
void mat_print(Mat m, const char *name);            //打印 m 中的内容
/* 访问矩阵第 i 行、第 j 列的元素，索引从 0 开始，可用于读取或赋值 */
#define MAT_PRINT(m) mat_print(m, #m)
int mat_dot(Mat dst, Mat a, Mat b);                 //将矩阵 a 乘 b 的结果输出到 dst 中
int mat_transpose(Mat dst, Mat src);                //将 src 的转置结果写入 dst
int mat_add_row(Mat dst, Mat a);                    //目标矩阵 dst 加上行向量 a
int mat_sgd_step(Mat param, Mat grad, float learning_rate);     //按 param -= learning_rate * grad 更新参数

int mat_sigmoid(Mat m);                             //对矩阵 m 的每个元素计算 Sigmoid，并将结果写回 m
int mat_relu(Mat m);                                //对矩阵 m 的每个元素计算 ReLU，并将结果写回 m

int mat_softmax(Mat m);                             //对矩阵 m 的每一行计算 Softmax，并将结果写回 m
size_t mat_argmax_row(Mat m, size_t row);           //返回矩阵 m 第 row 行最大元素的列索引，并列时取第一个
 
float rand_float(void);                             //生成 [0, 1] 范围内的随机浮点数

#endif //MATRIX_H_
