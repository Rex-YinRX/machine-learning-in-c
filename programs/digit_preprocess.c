#include "digit_preprocess.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

enum { SIDE = 28, PIXELS = SIDE * SIDE, TARGET = 20 };

static float brightness(float value)
{
    if (!isfinite(value) || value < 0.0f) {
        return 0.0f;
    }
    return value > 1.0f ? 1.0f : value;
}

/* Bilinear sampling; neighbors outside the image contribute zero. */
static inline float sample_bilinear(const float *src, float x, float y)
{
    if (src == NULL || !isfinite(x) || !isfinite(y) ||
        x <= -1.0f || x >= (float)SIDE ||
        y <= -1.0f || y >= (float)SIDE) {
        return 0.0f;
    }

    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    float fx = x - x0;
    float fy = y - y0;
    float sum = 0.0f;
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
            int col = x0 + i;
            int row = y0 + j;
            if (col >= 0 && col < SIDE && row >= 0 && row < SIDE) {
                float wx = i ? fx : 1.0f - fx;
                float wy = j ? fy : 1.0f - fy;
                sum += brightness(src[row * SIDE + col]) * wx * wy;
            }
        }
    }
    return brightness(sum);
}

/* Area averaging keeps narrow strokes when the crop is reduced. */
static float sample_area(const float *src, float x0, float y0,
                         float x1, float y1)
{
    float sum = 0.0f;
    for (int y = (int)floorf(y0); y < (int)ceilf(y1); ++y) {
        for (int x = (int)floorf(x0); x < (int)ceilf(x1); ++x) {
            if (x < 0 || x >= SIDE || y < 0 || y >= SIDE) {
                continue;
            }
            float wx = fminf(x1, (float)x + 1.0f) - fmaxf(x0, (float)x);
            float wy = fminf(y1, (float)y + 1.0f) - fmaxf(y0, (float)y);
            if (wx > 0.0f && wy > 0.0f) {
                sum += brightness(src[y * SIDE + x]) * wx * wy;
            }
        }
    }
    return brightness(sum / ((x1 - x0) * (y1 - y0)));
}

int digit_prepare(float *dst, const float *src)
{
    if (dst == NULL) {
        return 0;
    }
    if (src == NULL) {
        memset(dst, 0, PIXELS * sizeof(*dst));
        return 0;
    }

    int min_x = SIDE, min_y = SIDE, max_x = -1, max_y = -1;
    float input_mass = 0.0f;
    for (int y = 0; y < SIDE; ++y) {
        for (int x = 0; x < SIDE; ++x) {
            float value = brightness(src[y * SIDE + x]);
            input_mass += value;
            if (value > 0.02f) {
                if (x < min_x) min_x = x;
                if (x > max_x) max_x = x;
                if (y < min_y) min_y = y;
                if (y > max_y) max_y = y;
            }
        }
    }

    int crop_w = max_x - min_x + 1;
    int crop_h = max_y - min_y + 1;
    int longest = crop_w > crop_h ? crop_w : crop_h;
    if (max_x < 0 || longest < 4 || input_mass < 1.0f) {
        memset(dst, 0, PIXELS * sizeof(*dst));
        return 0;
    }

    /* Preserve aspect ratio; fit the foreground inside a 20x20 box. */
    int out_w = (int)roundf((float)crop_w * TARGET / longest);
    int out_h = (int)roundf((float)crop_h * TARGET / longest);
    if (out_w < 1) out_w = 1;
    if (out_h < 1) out_h = 1;
    int left = (SIDE - out_w) / 2;
    int top = (SIDE - out_h) / 2;
    float scaled[PIXELS] = {0};
    float mass = 0.0f, moment_x = 0.0f, moment_y = 0.0f;
    for (int y = 0; y < out_h; ++y) {
        for (int x = 0; x < out_w; ++x) {
            float x0 = min_x + (float)x * crop_w / out_w;
            float x1 = min_x + (float)(x + 1) * crop_w / out_w;
            float y0 = min_y + (float)y * crop_h / out_h;
            float y1 = min_y + (float)(y + 1) * crop_h / out_h;
            float value = sample_area(src, x0, y0, x1, y1);
            int col = left + x;
            int row = top + y;
            scaled[row * SIDE + col] = value;
            mass += value;
            moment_x += value * col;
            moment_y += value * row;
        }
    }

    if (mass <= 0.0f) {
        memset(dst, 0, PIXELS * sizeof(*dst));
        return 0;
    }

    /* Move the gray-level center of mass to the middle of the input. */
    float shift_x = 13.5f - moment_x / mass;
    float shift_y = 13.5f - moment_y / mass;
    /* Keep the whole crop in the image, even for asymmetric strokes. */
    shift_x = fmaxf(-(float)left,
                    fminf(shift_x, (float)(SIDE - left - out_w)));
    shift_y = fmaxf(-(float)top,
                    fminf(shift_y, (float)(SIDE - top - out_h)));
    float centered[PIXELS] = {0};
    for (int y = top; y < top + out_h; ++y) {
        for (int x = left; x < left + out_w; ++x) {
            float target_x = x + shift_x;
            float target_y = y + shift_y;
            int x0 = (int)floorf(target_x);
            int y0 = (int)floorf(target_y);
            float fx = target_x - x0;
            float fy = target_y - y0;
            for (int j = 0; j < 2; ++j) {
                for (int i = 0; i < 2; ++i) {
                    int col = x0 + i;
                    int row = y0 + j;
                    if (col >= 0 && col < SIDE && row >= 0 && row < SIDE) {
                        float wx = i ? fx : 1.0f - fx;
                        float wy = j ? fy : 1.0f - fy;
                        centered[row * SIDE + col] +=
                            scaled[y * SIDE + x] * wx * wy;
                    }
                }
            }
        }
    }
    for (int i = 0; i < PIXELS; ++i) {
        dst[i] = brightness(centered[i]);
    }
    return 1;
}

int digit_translate(float *dst, const float *src, int dx, int dy)
{
    if (dst == NULL || src == NULL || dst == src ||
        dx < -(SIDE - 1) || dx > SIDE - 1 ||
        dy < -(SIDE - 1) || dy > SIDE - 1) {
        return 0;
    }

    for (int y = 0; y < SIDE; ++y) {
        for (int x = 0; x < SIDE; ++x) {
            int source_x = x - dx;
            int source_y = y - dy;
            if (source_x >= 0 && source_x < SIDE &&
                source_y >= 0 && source_y < SIDE) {
                dst[y * SIDE + x] = src[source_y * SIDE + source_x];
            } else {
                dst[y * SIDE + x] = 0.0f;
            }
        }
    }
    return 1;
}

int digit_scale(float *dst, const float *src, float scale)
{
    if (dst == NULL || src == NULL || dst == src ||
        !isfinite(scale) || scale <= 0.0f) {
        return 0;
    }

    const float center = 13.5f;
    for (int y = 0; y < SIDE; ++y) {
        for (int x = 0; x < SIDE; ++x) {
            float source_x = center + (x - center) / scale;
            float source_y = center + (y - center) / scale;
            dst[y * SIDE + x] = sample_bilinear(src, source_x, source_y);
        }
    }
    return 1;
}

int digit_rotate(float *dst, const float *src, float degrees)
{
    if (dst == NULL || src == NULL || dst == src || !isfinite(degrees)) {
        return 0;
    }

    const float center = 13.5f;
    float radians = fmodf(degrees, 360.0f) * 0.017453292519943295f;
    float c = cosf(radians);
    float s = sinf(radians);
    for (int y = 0; y < SIDE; ++y) {
        for (int x = 0; x < SIDE; ++x) {
            float u = x - center;
            float v = y - center;
            /* Inverse mapping; positive angles look clockwise with y down. */
            float source_x = center + c * u + s * v;
            float source_y = center - s * u + c * v;
            dst[y * SIDE + x] = sample_bilinear(src, source_x, source_y);
        }
    }
    return 1;
}

int digit_augment(float *dst, const float *src, const DigitAugment *config)
{
    if (dst == NULL || src == NULL || config == NULL || dst == src) {
        return 0;
    }

    float scaled[PIXELS];
    float rotated[PIXELS];
    if (!digit_scale(scaled, src, config->scale)) {
        return 0;
    }
    if (!digit_rotate(rotated, scaled, config->degrees)) {
        return 0;
    }
    if (!digit_translate(dst, rotated, config->dx, config->dy)) {
        return 0;
    }
    return 1;
}

DigitAugment digit_random_config(void)
{
    DigitAugment config;
    config.scale = 0.9f + 0.2f * ((float)rand() / (float)RAND_MAX);
    config.degrees = -10.0f + 20.0f * ((float)rand() / (float)RAND_MAX);
    config.dx = rand() % 5 - 2;
    config.dy = rand() % 5 - 2;
    return config;
}
