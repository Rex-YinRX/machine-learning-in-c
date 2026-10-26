#include <stdio.h>
#include <stdint.h>

#include "matrix.h"

#define READ_TRAIN_IMG_COUNT 60000
#define READ_TEST_IMG_COUNT 10000

void read_32bit_uint(uint32_t *out, FILE *fp)
{
    unsigned char buffer[4];
    fread(buffer, 1, 4, fp); 
    uint32_t value =
            ((uint32_t)buffer[0] << 24) |
            ((uint32_t)buffer[1] << 16) |
            ((uint32_t)buffer[2] << 8)  |
            (uint32_t)buffer[3];
    *out = value;
}

void img_print(const uint8_t *pixels, size_t rows, size_t cols)
{
    for(size_t i = 0; i < rows; i++){
        for(size_t j = 0; j < cols; j++){
            printf("%3d ", pixels[i*cols + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    const char *data_dir = ".\\data";

    if(argc == 2){
        data_dir = argv[1];
    }else if(argc != 1){
        fprintf(stderr, "Usage: \"%s\" [data_dir]\n", argv[0]);
        return 1;
    }

    const char *file_names[6] = {
        "train-images.idx3-ubyte",
        "train-labels.idx1-ubyte",
        "t10k-images.idx3-ubyte",
        "t10k-labels.idx1-ubyte",
        "train\\train_data",
        "test\\test_data"
    };

    char paths[6][1024];

    for(size_t i = 0; i < 6; i++){
        int n = snprintf(paths[i],
            sizeof(paths[i]),
            "%s\\%s",
            data_dir,
            file_names[i]
        );

        if(n < 0){
            fprintf(stderr, "Path formatting failed\n");
            return 1;
        }

        if((size_t)n >= sizeof(paths[i])){
            fprintf(stderr, "Path too long: %s\\%s\n", data_dir, file_names[i]);
            return 1;
        }
    }
    
    FILE* img_fp = fopen(paths[0], "rb");
    if (img_fp == NULL) {
        perror(paths[0]);
        return 1;
    }
    FILE* label_fp = fopen(paths[1], "rb");
    if (label_fp == NULL) {
        perror(paths[1]);
        return 1;
    }

    FILE* ts_img_fp = fopen(paths[2], "rb");
    if (ts_img_fp == NULL) {
        perror(paths[2]);
        return 1;
    }
    FILE* ts_label_fp = fopen(paths[3], "rb");
    if (ts_label_fp == NULL) {
        perror(paths[3]);
        return 1;
    }

    uint32_t img_header[4];
    uint32_t lbl_header[2];
    uint32_t ts_img_header[4];
    uint32_t ts_lbl_header[2];
    uint8_t pixels[28 * 28];
    uint8_t labels[READ_TRAIN_IMG_COUNT];
    uint8_t ts_labels[READ_TEST_IMG_COUNT];

    //读取 train-images.idx3-ubyte 文件头
    for (size_t i = 0; i < (sizeof(img_header)/sizeof(img_header[0])); i++) {
        read_32bit_uint(&img_header[i], img_fp);
        //printf("%lu\n", (unsigned long)img_header[i]);
    }

    //读取 train-labels.idx1-ubyte 文件头
    for (size_t i = 0; i < (sizeof(lbl_header)/sizeof(lbl_header[0])); i++) {
        read_32bit_uint(&lbl_header[i], label_fp);
        //printf("%lu\n", (unsigned long)lbl_header[i]);
    }

    //读取 t10t-images.idx1-ubyte 文件头
    for (size_t i = 0; i < (sizeof(ts_img_header)/sizeof(ts_img_header[0])); i++) {
        read_32bit_uint(&ts_img_header[i], ts_img_fp);
        //printf("%lu\n", (unsigned long)img_header[i]);
    }

    //读取 t10t-labels.idx1-ubyte 文件头
    for (size_t i = 0; i < (sizeof(ts_lbl_header)/sizeof(ts_lbl_header[0])); i++) {
        read_32bit_uint(&ts_lbl_header[i], ts_label_fp);
        //printf("%lu\n", (unsigned long)lbl_header[i]);
    }

    
    Mat img = mat_alloc(READ_TRAIN_IMG_COUNT, 28*28 + 1);
    
    for(size_t i = 0; i < READ_TRAIN_IMG_COUNT; i++){
        fread(pixels, 1, 28*28, img_fp);
        //img_print(pixels, 28, 28);
        for(size_t j = 0; j < (sizeof(pixels)/sizeof(pixels[0])); j++){
            MAT_AT(img, i, j) = pixels[j]/255.0f;
        }
        fread(&labels[i], sizeof(labels[0]), 1, label_fp);
        MAT_AT(img, i, 28*28) = (float)labels[i];
    }
    //MAT_PRINT(img);

    FILE* out = fopen(paths[4], "wb");
    if (out == NULL) {
        perror(paths[4]);
        return 1;
    }
    mat_save(out, img);
    printf("train data saved\n");

    Mat ts_img = mat_alloc(READ_TEST_IMG_COUNT, 28*28 + 1);

    for(size_t i = 0; i < READ_TEST_IMG_COUNT; i++){
        fread(pixels, 1, 28*28, ts_img_fp);
        //img_print(pixels, 28, 28);
        for(size_t j = 0; j < (sizeof(pixels)/sizeof(pixels[0])); j++){
            MAT_AT(ts_img, i, j) = pixels[j]/255.0f;
        }
        fread(&ts_labels[i], sizeof(ts_labels[0]), 1, ts_label_fp);
        MAT_AT(ts_img, i, 28*28) = (float)ts_labels[i];
    }
    //MAT_PRINT(ts_img);

    FILE* ts_out = fopen(paths[5], "wb");
    if (ts_out == NULL) {
        perror(paths[5]);
        return 1;
    }
    mat_save(ts_out, ts_img);
    printf("test data saved\n");


    printf("mission finished\n");

    mat_free(&img);
    mat_free(&ts_img);
    fclose(img_fp);
    fclose(label_fp);
    fclose(out);
    fclose(ts_img_fp);
    fclose(ts_label_fp);
    fclose(ts_out);
    return 0;
}