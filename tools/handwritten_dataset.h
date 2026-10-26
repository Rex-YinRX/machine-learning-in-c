#ifndef HANDWRITTEN_DATASET_H_
#define HANDWRITTEN_DATASET_H_

#include "matrix.h"

enum {
    HANDWRITTEN_PIXELS = 784,
    HANDWRITTEN_COLUMNS = 785,
    HANDWRITTEN_MAX_AUGMENT = 100,
    HANDWRITTEN_PATH_CAP = 1024
};

typedef struct {
    size_t id;
    size_t rows;
    char raw_path[HANDWRITTEN_PATH_CAP];
    char augmented_path[HANDWRITTEN_PATH_CAP];
} HandwrittenSaveInfo;

typedef struct {
    size_t augment_count;
    unsigned int seed;
} HandwrittenExportOptions;

typedef struct {
    size_t raw_samples;
    size_t rows;
    char output_dir[HANDWRITTEN_PATH_CAP];
    char dataset_path[HANDWRITTEN_PATH_CAP];
    char info_path[HANDWRITTEN_PATH_CAP];
} HandwrittenExportInfo;

/* Filesystem helpers run without a window. root is normally data/handwritten. */
int handwritten_ensure_dirs(const char *root);

/* Owned (augment_count+1)x785 matrix, or empty; raw input remains unchanged. */
Mat handwritten_make_samples(const float *raw_pixels, int label, size_t augment_count);

/* 1 = both saved, 2 = raw saved but augmentation failed, 0 = no raw saved.
 * IDs skip existing files in both directories. Never overwrite samples.
 * info receives the paths/ID, including the raw-only case. */
int handwritten_save_sample(const char *root, const float *raw_pixels, int label,
                            size_t augment_count, size_t *next_id,
                            HandwrittenSaveInfo *info);

/* Merge all raw samples into one dataset.cmat in a NEW root/datasets/NNNNNN.
 * Each source contributes its prepared image and augment_count variants.
 * No training/validation split. Existing samples and exports are preserved.
 * dataset_info.txt is written last to mark completion. */
int handwritten_export_dataset(const char *root,
                               const HandwrittenExportOptions *options,
                               HandwrittenExportInfo *info);

#endif