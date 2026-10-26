#include "handwritten_dataset.h"
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_unsigned(const char *text, unsigned int maximum, unsigned int *result)
{
    if (text == NULL || text[0] < '0' || text[0] > '9') return 0;
    char *end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno != 0 || *end != '\0' || value > maximum) return 0;
    *result = (unsigned int)value;
    return 1;
}

static void usage(const char *program)
{
    printf("Usage: \"%s\" [data_dir [augment_count [seed]]]\n", program);
    printf("Defaults: data/handwritten, 20, 1234.\n");
    printf("augment_count: 0..%d.\n", HANDWRITTEN_MAX_AUGMENT);
    printf("Reads all raw/*.cmat and exports a new datasets/NNNNNN directory.\n");
    printf("dataset.cmat contains one prepared original plus its variants per source.\n");
    printf("No model or graphical window is needed.\n");
}

int main(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        usage(argv[0]);
        return 0;
    }
    if (argc > 4 || (argc >= 2 && argv[1][0] == '-')) {
        usage(argv[0]);
        return 1;
    }
    const char *data_root = argc >= 2 ? argv[1] : "data/handwritten";
    if (!data_root[0]) {
        fprintf(stderr, "data_dir must not be empty.\n");
        return 1;
    }
    HandwrittenExportOptions options = {20, 1234u};
    unsigned int value = 0;
    if (argc >= 3) {
        if (!parse_unsigned(argv[2], HANDWRITTEN_MAX_AUGMENT, &value)) {
            fprintf(stderr, "augment_count must be an integer from 0 to %d.\n", HANDWRITTEN_MAX_AUGMENT);
            return 1;
        }
        options.augment_count = value;
    }
    if (argc >= 4) {
        if (!parse_unsigned(argv[3], UINT_MAX, &value)) {
            fprintf(stderr, "seed must be an unsigned integer no greater than %u.\n", UINT_MAX);
            return 1;
        }
        options.seed = value;
    }
    HandwrittenExportInfo info = {0};
    if (!handwritten_export_dataset(data_root, &options, &info)) return 1;
    printf("Dataset directory: %s\n", info.output_dir);
    printf("Original sources: %zu\n", info.raw_samples);
    printf("Dataset: %s (%zu rows x %d columns)\n", info.dataset_path,
           info.rows, HANDWRITTEN_COLUMNS);
    printf("Dataset info: %s\n", info.info_path);
    printf("Seed: %u; variants per original: %zu\n", options.seed, options.augment_count);
    return 0;
}