#include "handwritten_dataset.h"
#include "digit_preprocess.h"
#include "raylib.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

typedef struct {
    const char *path;
    int label;
} SourceRecord;

static int path_join(char *dst, size_t capacity, const char *root, const char *tail)
{
    if (dst == NULL || capacity == 0 || root == NULL || root[0] == '\0' || tail == NULL) return 0;
    size_t length = strlen(root);
    int separator = root[length - 1] != '/' && root[length - 1] != '\\';
    int written = snprintf(dst, capacity, "%s%s%s", root, separator ? "/" : "", tail);
    if (written < 0 || (size_t)written >= capacity) {
        fprintf(stderr, "Handwritten data path is too long.\n");
        return 0;
    }
    return 1;
}

static int path_exists(const char *path)
{
    return FileExists(path) || DirectoryExists(path);
}

static int ensure_directory(const char *path)
{
    if (DirectoryExists(path)) return 1;
    if (FileExists(path) || MakeDirectory(path) != 0) {
        fprintf(stderr, "Cannot create directory: %s\n", path);
        return 0;
    }
    return 1;
}

int handwritten_ensure_dirs(const char *root)
{
    if (root == NULL || root[0] == '\0' || strlen(root) >= HANDWRITTEN_PATH_CAP) {
        fprintf(stderr, "Handwritten data directory is empty or too long.\n");
        return 0;
    }
    const char *parts[] = {"raw", "augmented", "datasets"};
    char path[HANDWRITTEN_PATH_CAP];
    if (!ensure_directory(root)) return 0;
    for (size_t i = 0; i < sizeof parts / sizeof parts[0]; ++i) {
        if (!path_join(path, sizeof path, root, parts[i]) || !ensure_directory(path)) return 0;
    }
    return 1;
}

static int valid_pixels(const float *pixels)
{
    if (pixels == NULL) return 0;
    for (size_t i = 0; i < HANDWRITTEN_PIXELS; ++i) {
        if (!isfinite(pixels[i]) || pixels[i] < 0.0f || pixels[i] > 1.0f) return 0;
    }
    return 1;
}

Mat handwritten_make_samples(const float *raw_pixels, int label, size_t augment_count)
{
    Mat empty = {0};
    if (!valid_pixels(raw_pixels) || label < 0 || label > 9 ||
        augment_count > HANDWRITTEN_MAX_AUGMENT) return empty;
    float prepared[HANDWRITTEN_PIXELS];
    if (!digit_prepare(prepared, raw_pixels)) return empty;
    Mat batch = mat_alloc(augment_count + 1, HANDWRITTEN_COLUMNS);
    if (batch.es == NULL) return empty;
    memcpy(batch.es, prepared, sizeof prepared);
    MAT_AT(batch, 0, HANDWRITTEN_PIXELS) = (float)label;
    for (size_t row = 1; row < batch.rows; ++row) {
        DigitAugment config = digit_random_config();
        if (!digit_augment(batch.es + row * batch.stride, prepared, &config)) {
            mat_free(&batch);
            return empty;
        }
        MAT_AT(batch, row, HANDWRITTEN_PIXELS) = (float)label;
    }
    return batch;
}

/* Close every created file, and discard only that new file on failure. */
static int finish_file(FILE *out, const char *path)
{
    int ok = !ferror(out);
    if (fclose(out) != 0) ok = 0;
    if (!ok) {
        fprintf(stderr, "Incomplete write: %s\n", path);
        if (remove(path) != 0) fprintf(stderr, "Cannot remove incomplete file: %s\n", path);
    }
    return ok;
}

static int write_matrix_exclusive(const char *path, Mat matrix)
{
    FILE *out = fopen(path, "wbx");
    if (out == NULL) {
        perror(path);
        return 0;
    }
    mat_save(out, matrix);
    return finish_file(out, path);
}

int handwritten_save_sample(const char *root, const float *raw_pixels, int label,
                            size_t augment_count, size_t *next_id,
                            HandwrittenSaveInfo *info)
{
    if (info != NULL) memset(info, 0, sizeof *info);
    if (next_id == NULL || info == NULL || !valid_pixels(raw_pixels) ||
        label < 0 || label > 9 || augment_count > HANDWRITTEN_MAX_AUGMENT) {
        fprintf(stderr, "Choose a label 0-9 and use a valid image/augmentation count.\n");
        return 0;
    }
    float prepared[HANDWRITTEN_PIXELS];
    if (!digit_prepare(prepared, raw_pixels)) {
        fprintf(stderr, "Draw a digit first; blank or tiny strokes are not saved.\n");
        return 0;
    }
    if (!handwritten_ensure_dirs(root)) return 0;
    size_t candidate = *next_id == 0 ? 1 : *next_id;
    HandwrittenSaveInfo result = {0};
    for (;;) {
        char raw_tail[96], augmented_tail[96];
        int a = snprintf(raw_tail, sizeof raw_tail, "raw/%06zu.cmat", candidate);
        int b = snprintf(augmented_tail, sizeof augmented_tail, "augmented/%06zu.cmat", candidate);
        if (a < 0 || (size_t)a >= sizeof raw_tail || b < 0 || (size_t)b >= sizeof augmented_tail ||
            !path_join(result.raw_path, sizeof result.raw_path, root, raw_tail) ||
            !path_join(result.augmented_path, sizeof result.augmented_path, root, augmented_tail)) return 0;
        if (!path_exists(result.raw_path) && !path_exists(result.augmented_path)) break;
        if (candidate == SIZE_MAX) {
            fprintf(stderr, "No unused handwritten sample ID remains.\n");
            return 0;
        }
        ++candidate;
    }
    float row[HANDWRITTEN_COLUMNS];
    memcpy(row, raw_pixels, HANDWRITTEN_PIXELS * sizeof *row);
    row[HANDWRITTEN_PIXELS] = (float)label;
    Mat raw = {1, HANDWRITTEN_COLUMNS, HANDWRITTEN_COLUMNS, row};
    if (!write_matrix_exclusive(result.raw_path, raw)) return 0;
    result.id = candidate;
    result.rows = augment_count + 1;
    *info = result;
    *next_id = candidate == SIZE_MAX ? SIZE_MAX : candidate + 1;
    Mat batch = handwritten_make_samples(raw_pixels, label, augment_count);
    if (batch.es == NULL) {
        fprintf(stderr, "Raw sample saved, but augmented batch could not be generated: %s\n", result.raw_path);
        return 2;
    }
    int ok = write_matrix_exclusive(result.augmented_path, batch);
    mat_free(&batch);
    if (!ok) {
        fprintf(stderr, "Raw sample saved, but augmentation save failed: %s\n", result.raw_path);
        return 2;
    }
    return 1;
}

/* Read only the expected fixed shape; reject bad headers before allocating. */
static Mat read_raw_sample(const char *path, int *label)
{
    Mat empty = {0};
    FILE *in = fopen(path, "rb");
    if (in == NULL) { perror(path); return empty; }
    char magic[4];
    size_t rows = 0, columns = 0;
    float values[HANDWRITTEN_COLUMNS];
    int ok = fread(magic, 1, sizeof magic, in) == sizeof magic &&
             memcmp(magic, "CMAT", sizeof magic) == 0 &&
             fread(&rows, sizeof rows, 1, in) == 1 &&
             fread(&columns, sizeof columns, 1, in) == 1 &&
             rows == 1 && columns == HANDWRITTEN_COLUMNS;
    if (ok) {
        ok = fread(values, sizeof *values, HANDWRITTEN_COLUMNS, in) == HANDWRITTEN_COLUMNS;
        if (ok) ok = fgetc(in) == EOF && !ferror(in);
    }
    if (fclose(in) != 0) ok = 0;
    if (ok) {
        float value = values[HANDWRITTEN_PIXELS];
        ok = valid_pixels(values) && isfinite(value) && value >= 0.0f && value <= 9.0f &&
             value == floorf(value);
    }
    float prepared[HANDWRITTEN_PIXELS];
    if (ok) ok = digit_prepare(prepared, values);
    if (!ok) {
        fprintf(stderr, "Invalid raw sample (expected one valid 1x785 CMAT row): %s\n", path);
        return empty;
    }
    Mat sample = mat_alloc(1, HANDWRITTEN_COLUMNS);
    if (sample.es == NULL) {
        fprintf(stderr, "Cannot allocate raw sample: %s\n", path);
        return empty;
    }
    memcpy(sample.es, values, sizeof values);
    *label = (int)values[HANDWRITTEN_PIXELS];
    return sample;
}

static int compare_paths(const void *a, const void *b)
{
    const char *const *left = a;
    const char *const *right = b;
    return strcmp(*left, *right);
}

/* Atomically claim a new snapshot directory, with its parent already present. */
static int new_snapshot_directory(const char *root, char *output, size_t capacity)
{
    for (size_t id = 1; ; ++id) {
        char tail[96];
        int written = snprintf(tail, sizeof tail, "datasets/%06zu", id);
        if (written < 0 || (size_t)written >= sizeof tail || !path_join(output, capacity, root, tail)) return 0;
        if (!path_exists(output)) {
#ifdef _WIN32
            if (_mkdir(output) == 0) return 1;
#else
            if (mkdir(output, 0777) == 0) return 1;
#endif
            if (errno != EEXIST) { perror(output); return 0; }
        }
        if (id == SIZE_MAX) {
            fprintf(stderr, "No unused dataset snapshot ID remains.\n");
            return 0;
        }
    }
}

static int write_summary(const char *path, const HandwrittenExportOptions *options,
                         const HandwrittenExportInfo *info, const size_t label_counts[10])
{
    FILE *out = fopen(path, "wbx");
    if (out == NULL) { perror(path); return 0; }
    int ok = fprintf(out,
        "format=CMAT native size_t/float\ncolumns=785\nraw_samples=%zu\nrows=%zu\n"
        "augment_count=%zu\nseed=%u\n"
        "source_order=filename ascending\nrows_per_source=%zu\n"
        "scale_range=0.9,1.1\nrotation_degrees=-10,10\ntranslation_pixels=-2,2\n"
        "dataset=dataset.cmat\n",
        info->raw_samples, info->rows, options->augment_count, options->seed,
        options->augment_count + 1) >= 0;
    for (int i = 0; i < 10 && ok; ++i)
        ok = fprintf(out, "label_%d_sources=%zu\n", i, label_counts[i]) >= 0;
    if (!ok) {
        fclose(out);
        if (remove(path) != 0) fprintf(stderr, "Cannot remove incomplete summary: %s\n", path);
        return 0;
    }
    return finish_file(out, path);
}

int handwritten_export_dataset(const char *root,
                               const HandwrittenExportOptions *options,
                               HandwrittenExportInfo *info)
{
    if (info != NULL) memset(info, 0, sizeof *info);
    if (info == NULL || options == NULL || options->augment_count > HANDWRITTEN_MAX_AUGMENT) {
        fprintf(stderr, "Invalid export options (augmentation count must be 0-100).\n");
        return 0;
    }
    if (!handwritten_ensure_dirs(root)) return 0;
    char raw_directory[HANDWRITTEN_PATH_CAP];
    if (!path_join(raw_directory, sizeof raw_directory, root, "raw")) return 0;
    FilePathList files = LoadDirectoryFilesEx(raw_directory, ".cmat", false);
    if (files.count == 0) {
        fprintf(stderr, "No raw samples found in %s. Draw, label, and save samples first.\n", raw_directory);
        UnloadDirectoryFiles(files);
        return 0;
    }
    qsort(files.paths, files.count, sizeof *files.paths, compare_paths);
    size_t count = files.count;
    if (count > SIZE_MAX / sizeof(SourceRecord)) {
        UnloadDirectoryFiles(files);
        return 0;
    }
    SourceRecord *sources = calloc(count, sizeof *sources);
    Mat dataset = {0};
    HandwrittenExportInfo result = {0};
    size_t label_counts[10] = {0};
    int success = 0;
    if (sources == NULL) {
        fprintf(stderr, "Cannot allocate source list.\n");
        goto cleanup;
    }
    /* Validate every source before creating any export snapshot. */
    for (size_t i = 0; i < count; ++i) {
        if (strlen(files.paths[i]) >= HANDWRITTEN_PATH_CAP) {
            fprintf(stderr, "Raw sample path is too long.\n");
            goto cleanup;
        }
        sources[i].path = files.paths[i];
        Mat raw = read_raw_sample(sources[i].path, &sources[i].label);
        if (raw.es == NULL) goto cleanup;
        mat_free(&raw);
        ++label_counts[sources[i].label];
    }
    size_t rows_per_source = options->augment_count + 1;
    if (count > SIZE_MAX / rows_per_source) {
        fprintf(stderr, "Dataset row count is too large.\n");
        goto cleanup;
    }
    size_t total_rows = count * rows_per_source;
    if (total_rows > SIZE_MAX / HANDWRITTEN_COLUMNS / sizeof(float)) {
        fprintf(stderr, "Dataset matrix dimensions are too large.\n");
        goto cleanup;
    }
    dataset = mat_alloc(total_rows, HANDWRITTEN_COLUMNS);
    if (dataset.es == NULL) {
        fprintf(stderr, "Cannot allocate output dataset.\n");
        goto cleanup;
    }
    /* Stable source order and one seed make repeated exports reproducible. */
    srand(options->seed);
    for (size_t index = 0; index < count; ++index) {
        SourceRecord *source = &sources[index];
        int actual_label = -1;
        Mat raw = read_raw_sample(source->path, &actual_label);
        if (raw.es == NULL) goto cleanup;
        if (actual_label != source->label) {
            fprintf(stderr, "Raw sample changed during export: %s\n", source->path);
            mat_free(&raw);
            goto cleanup;
        }
        Mat batch = handwritten_make_samples(raw.es, source->label, options->augment_count);
        mat_free(&raw);
        if (batch.es == NULL) {
            fprintf(stderr, "Cannot generate samples from: %s\n", source->path);
            goto cleanup;
        }
        size_t start_row = index * rows_per_source;
        for (size_t row = 0; row < batch.rows; ++row)
            memcpy(dataset.es + (start_row + row) * dataset.stride,
                   batch.es + row * batch.stride, HANDWRITTEN_COLUMNS * sizeof(float));
        mat_free(&batch);
    }
    result.raw_samples = count;
    result.rows = total_rows;
    if (!new_snapshot_directory(root, result.output_dir, sizeof result.output_dir)) goto cleanup;
    if (!path_join(result.dataset_path, sizeof result.dataset_path, result.output_dir, "dataset.cmat") ||
        !path_join(result.info_path, sizeof result.info_path, result.output_dir, "dataset_info.txt")) goto cleanup;
    if (!write_matrix_exclusive(result.dataset_path, dataset)) goto cleanup;
    /* The summary is the completion marker and is always written last. */
    if (!write_summary(result.info_path, options, &result, label_counts)) goto cleanup;
    *info = result;
    success = 1;
cleanup:
    if (!success && result.output_dir[0] != '\0')
        fprintf(stderr, "Export did not complete: %s (no complete dataset_info.txt).\n", result.output_dir);
    mat_free(&dataset);
    free(sources);
    UnloadDirectoryFiles(files);
    return success;
}