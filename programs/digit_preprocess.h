#ifndef DIGIT_PREPROCESS_H_
#define DIGIT_PREPROCESS_H_

int digit_prepare(float *dst, const float *src);

/*
 * Translate separate 784-float, row-major 28x28 buffers.
 * Positive dx/dy moves right/down. Cropped pixels are discarded; gaps are zero.
 * dx and dy must be in [-27, 27]. Success completely overwrites dst.
 * Return 0 without writing for NULL pointers, dst == src, or invalid offsets.
 */
int digit_translate(float *dst, const float *src, int dx, int dy);

/*
 * Scale separate 784-float, row-major 28x28 buffers about (13.5, 13.5).
 * Use bilinear sampling and zero padding; output remains 28x28.
 * scale must be finite and positive: >1 enlarges, <1 shrinks.
 * Success fully overwrites dst and preserves src.
 * Return 0 without writing for NULL pointers, dst == src, or invalid scale.
 */
int digit_scale(float *dst, const float *src, float scale);

/*
 * Rotate separate 784-float, row-major 28x28 buffers about (13.5, 13.5).
 * degrees is finite, in degrees: positive clockwise, negative counterclockwise.
 * Zero degrees is valid. Output remains 28x28, using bilinear zero padding.
 * Success fully overwrites dst and preserves src.
 * Return 0 without writing for NULL pointers, dst == src, or nonfinite degrees.
 */
int digit_rotate(float *dst, const float *src, float degrees);

typedef struct {
    float scale;
    float degrees;
    int dx;
    int dy;
} DigitAugment;

/*
 * Generate scale in [0.9, 1.1], degrees in [-10, 10], dx/dy in [-2, 2].
 * Each parameter uses a separate rand() call. The caller controls srand().
 */
DigitAugment digit_random_config(void);

/*
 * Apply scale -> clockwise rotation -> translation to a prepared 28x28 image.
 * src and dst must be separate 784-float buffers; src/config are preserved.
 * Parameter ranges follow digit_scale, digit_rotate, and digit_translate.
 * Return 1 on success, fully overwriting dst; return 0 without changing dst.
 * No preparation or label processing is performed here.
 */
int digit_augment(float *dst, const float *src, const DigitAugment *config);

#endif
