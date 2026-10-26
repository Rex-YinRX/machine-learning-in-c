#include "raylib.h"
#include "network.h"
#include "digit_preprocess.h"
#include "handwritten_dataset.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void canvas_brush(float *pixels, int col, int row)
{
    const float radius = 1.5f;
    for (int y = row - 2; y <= row + 2; ++y) {
        for (int x = col - 2; x <= col + 2; ++x) {
            if (x < 0 || x >= 28 || y < 0 || y >= 28) continue;
            float dx = (float)(x - col), dy = (float)(y - row);
            float distance = sqrtf(dx * dx + dy * dy);
            if (distance >= radius) continue;
            float strength = 1.0f - distance / radius;
            if (strength > pixels[y * 28 + x]) pixels[y * 28 + x] = strength;
        }
    }
}

void canvas_line(float *pixels, int x0, int y0, int x1, int y1)
{
    int dx = x1 - x0, dy = y1 - y0;
    int steps = abs(dx);
    if (abs(dy) > steps) steps = abs(dy);
    if (steps == 0) {
        canvas_brush(pixels, x0, y0);
        return;
    }
    for (int k = 0; k <= steps; ++k) {
        float t = (float)k / steps;
        canvas_brush(pixels, (int)roundf(x0 + dx * t), (int)roundf(y0 + dy * t));
    }
}

static int ui_scaled_size(int base, float scale, int minimum)
{
    int size = (int)roundf(base * scale);
    return size < minimum ? minimum : size;
}

static void draw_fit_text(const char *text, int x, int y, int width,
                          int font_size, Color color)
{
    while (font_size > 10 && MeasureText(text, font_size) > width) --font_size;
    DrawText(text, x, y, font_size, color);
}

static int parse_augment_count(const char *text, size_t *count)
{
    if (text == NULL || text[0] < '0' || text[0] > '9') return 0;
    char *end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno != 0 || *end != '\0' || value > HANDWRITTEN_MAX_AUGMENT) return 0;
    *count = (size_t)value;
    return 1;
}

static void print_usage(const char *program)
{
    printf("Usage: \"%s\" [model_path [data_dir [augment_count]]]\n", program);
    printf("Defaults: model/mnist_model.bin, data/handwritten, 20 variants.\n");
    printf("0-9: choose true label; S: save raw and variants; E: export dataset; C: clear.\n");
    printf("E exports all saved originals with seed 1234.\n");
}

int main(int argc, char *argv[])
{
    const char *model_path = "model/mnist_model.bin";
    const char *data_root = "data/handwritten";
    size_t augment_count = 20;
    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        print_usage(argv[0]);
        return 0;
    }
    if (argc > 4) {
        print_usage(argv[0]);
        return 1;
    }
    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) data_root = argv[2];
    if (argc >= 4 && !parse_augment_count(argv[3], &augment_count)) {
        fprintf(stderr, "augment_count must be an integer from 0 to %d.\n",
                HANDWRITTEN_MAX_AUGMENT);
        return 1;
    }

    FILE *model = fopen(model_path, "rb");
    if (model == NULL) {
        perror(model_path);
        return 1;
    }
    Network net = network_load(model);
    fclose(model);
    if (net.count == 0) {
        fprintf(stderr, "Could not load model: %s\n", model_path);
        network_free(&net);
        return 1;
    }
    if (!handwritten_ensure_dirs(data_root)) {
        network_free(&net);
        return 1;
    }
    Mat scores = mat_alloc(1, 10);
    if (scores.es == NULL) {
        fprintf(stderr, "Could not allocate prediction output.\n");
        network_free(&net);
        return 1;
    }
    srand((unsigned int)time(NULL));
    printf("Data directory: %s\nVariants per original: %zu\n", data_root, augment_count);
    printf("S saves raw + augmented samples; E exports all raw samples.\n");
    fflush(stdout);

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(800, 450, "Digit Collector");
    SetWindowMinSize(320, 260);
    SetTargetFPS(60);
    float pixels[784] = {0}, model_pixels[784] = {0};
    Mat input = {.rows = 1, .cols = 784, .stride = 784, .es = model_pixels};
    int prev_col = 0, prev_row = 0, has_previous = 0;
    int prediction = -1, label = -1;
    size_t next_id = 1;
    char status[256] = "Ready: draw, choose 0-9, then S to save";
    Color status_color = DARKGRAY;

    while (!WindowShouldClose()) {
        int need_predict = 0;
        int width = GetScreenWidth(), height = GetScreenHeight();
        float ui_scale = fminf((float)width / 800.0f, (float)height / 450.0f);
        int padding = ui_scaled_size(20, ui_scale, 8);
        int gap = ui_scaled_size(8, ui_scale, 6);
        int bottom_padding = ui_scaled_size(16, ui_scale, 8);
        int prediction_font = ui_scaled_size(30, ui_scale, 18);
        int hint_font = ui_scaled_size(20, ui_scale, 14);
        int title_font = ui_scaled_size(18, ui_scale, 12);
        int helper_font = ui_scaled_size(16, ui_scale, 12);
        int preview_size = ui_scaled_size(112, ui_scale, 28);
        int text_top = padding / 2;
        int top_font = prediction_font > hint_font ? prediction_font : hint_font;
        int status_y = text_top + top_font + gap;
        int canvas_top = status_y + helper_font + gap;
        int footer_content = 3 * helper_font + 2 * gap;
        if (footer_content < preview_size) footer_content = preview_size;
        int footer_height = footer_content + title_font + 2 * gap + bottom_padding;
        int available_width = width - 2 * padding;
        int available_height = height - canvas_top - footer_height;
        int available = available_width < available_height ? available_width : available_height;
        int pixel_size = available / 28;
        if (pixel_size < 1) pixel_size = 1;
        int canvas_size = pixel_size * 28;
        int canvas_x = (width - canvas_size) / 2;
        int canvas_y = canvas_top + (available_height - canvas_size) / 2;
        Vector2 mouse = GetMousePosition();
        if (IsWindowResized()) has_previous = 0;

        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) &&
            mouse.x >= canvas_x && mouse.x < canvas_x + canvas_size &&
            mouse.y >= canvas_y && mouse.y < canvas_y + canvas_size) {
            int col = (int)((mouse.x - canvas_x) * 28 / canvas_size);
            int row = (int)((mouse.y - canvas_y) * 28 / canvas_size);
            if (has_previous) canvas_line(pixels, prev_col, prev_row, col, row);
            else canvas_brush(pixels, col, row);
            prev_col = col; prev_row = row; has_previous = 1;
            need_predict = 1;
        } else {
            has_previous = 0;
        }
        int ch = GetCharPressed();
        while (ch != 0) {
            if (ch >= '0' && ch <= '9') label = ch - '0';
            ch = GetCharPressed();
        }
        if (IsKeyPressed(KEY_C)) {
            memset(pixels, 0, sizeof pixels);
            memset(model_pixels, 0, sizeof model_pixels);
            has_previous = 0; prediction = -1; label = -1; need_predict = 0;
            snprintf(status, sizeof status, "Ready: draw, choose 0-9, then S to save");
            status_color = DARKGRAY;
        }
        if (need_predict) {
            prediction = digit_prepare(model_pixels, pixels) && network_forward(scores, &net, input)
                       ? (int)mat_argmax_row(scores, 0) : -1;
        }
        if (IsKeyPressed(KEY_S)) {
            HandwrittenSaveInfo info = {0};
            int result = handwritten_save_sample(data_root, pixels, label, augment_count,
                                                  &next_id, &info);
            if (result == 1) {
                snprintf(status, sizeof status, "Saved #%06zu: %zu rows. C for next.", info.id, info.rows);
                status_color = DARKGREEN;
                printf("Saved raw sample: %s (label=%d)\n", info.raw_path, label);
                printf("Saved augmented samples: %s (%zu rows)\n", info.augmented_path, info.rows);
            } else if (result == 2) {
                snprintf(status, sizeof status, "Raw #%06zu saved; variants failed", info.id);
                status_color = ORANGE;
                fprintf(stderr, "Raw sample preserved: %s\nAugmentation failed: %s\n",
                        info.raw_path, info.augmented_path);
            } else {
                snprintf(status, sizeof status, "Save failed: check label, drawing, console");
                status_color = MAROON;
            }
            fflush(stdout);
        }
        if (IsKeyPressed(KEY_E)) {
            HandwrittenExportOptions options = {augment_count, 1234u};
            HandwrittenExportInfo info = {0};
            if (handwritten_export_dataset(data_root, &options, &info)) {
                snprintf(status, sizeof status, "Exported: %zu originals, %zu rows",
                         info.raw_samples, info.rows);
                status_color = DARKGREEN;
                printf("Dataset exported: %s\nDataset: %s (%zu rows)\n", info.output_dir,
                       info.dataset_path, info.rows);
                printf("Dataset info: %s\n", info.info_path);
            } else {
                snprintf(status, sizeof status, "Export failed: check saved raw samples/console");
                status_color = MAROON;
            }
            fflush(stdout);
        }

        BeginDrawing();
        ClearBackground(LIGHTGRAY);
        DrawRectangle(canvas_x, canvas_y, canvas_size, canvas_size, BLACK);
        for (int row = 0; row < 28; ++row) {
            for (int col = 0; col < 28; ++col) {
                float value = pixels[row * 28 + col];
                if (value > 0.0f) {
                    unsigned char gray = (unsigned char)(value * 255.0f);
                    Color color = {gray, gray, gray, 255};
                    DrawRectangle(canvas_x + col * pixel_size, canvas_y + row * pixel_size,
                                  pixel_size, pixel_size, color);
                }
            }
        }
        if (prediction >= 0)
            draw_fit_text(TextFormat("Prediction: %d", prediction), padding, text_top,
                          width - 2 * padding, prediction_font, DARKGREEN);
        else
            draw_fit_text("Draw a digit", padding, text_top, width - 2 * padding, hint_font, DARKGRAY);
        draw_fit_text(status, padding, status_y, width - 2 * padding, helper_font, status_color);
        int footer_top = height - bottom_padding - footer_content;
        int preview_x = width - preview_size - padding;
        int preview_y = height - preview_size - bottom_padding;
        int title_x = width - padding - MeasureText("Model input", title_font);
        DrawText("Model input", title_x, footer_top - title_font - gap, title_font, DARKGRAY);
        DrawRectangle(preview_x, preview_y, preview_size, preview_size, BLACK);
        for (int row = 0; row < 28; ++row) {
            for (int col = 0; col < 28; ++col) {
                unsigned char gray = (unsigned char)(model_pixels[row * 28 + col] * 255.0f);
                Color color = {gray, gray, gray, 255};
                int x0 = preview_x + col * preview_size / 28;
                int x1 = preview_x + (col + 1) * preview_size / 28;
                int y0 = preview_y + row * preview_size / 28;
                int y1 = preview_y + (row + 1) * preview_size / 28;
                DrawRectangle(x0, y0, x1 - x0, y1 - y0, color);
            }
        }
        int helper_width = preview_x - padding - gap;
        draw_fit_text(TextFormat("S: save %zu rows", augment_count + 1), padding, footer_top,
                      helper_width, helper_font, DARKGRAY);
        draw_fit_text("E: export dataset", padding, footer_top + helper_font + gap,
                      helper_width, helper_font, DARKGRAY);
        const char *label_text = label >= 0 ? TextFormat("Label: %d | 0-9: change | C: clear", label)
                                            : "0-9: set label | C: clear";
        draw_fit_text(label_text, padding, footer_top + 2 * (helper_font + gap),
                      helper_width, helper_font, label >= 0 ? DARKGREEN : DARKGRAY);
        EndDrawing();
    }
    CloseWindow();
    mat_free(&scores);
    network_free(&net);
    return 0;
}