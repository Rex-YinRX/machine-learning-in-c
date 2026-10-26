#include "raylib.h"
#include "network.h"
#include "digit_preprocess.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void canvas_brush(float *pixels, int col, int row)
{
    const float radius = 1.5f;

    for (int y = row - 2; y <= row + 2; ++y) {
        for (int x = col - 2; x <= col + 2; ++x) {
            if (x < 0 || x >= 28 || y < 0 || y >= 28) {
                continue;
            }

            float dx = (float)(x - col);
            float dy = (float)(y - row);
            float distance = sqrtf(dx * dx + dy * dy);
            if (distance >= radius) {
                continue;
            }

            float strength = 1.0f - distance / radius;
            if (strength > pixels[y * 28 + x]) {
                pixels[y * 28 + x] = strength;
            }
        }
    }
}

void canvas_line(float *pixels, int x0, int y0, int x1, int y1)
{
    int dx = x1 - x0;
    int dy = y1 - y0;
    int steps = abs(dx);
    if (abs(dy) > steps) {
        steps = abs(dy);
    }

    if (steps == 0) {
        canvas_brush(pixels, x0, y0);
        return;
    }

    for (int k = 0; k <= steps; ++k) {
        float t = (float)k / steps;
        int col = (int)roundf(x0 + dx * t);
        int row = (int)roundf(y0 + dy * t);
        canvas_brush(pixels, col, row);
    }
}

static int ui_scaled_size(int base, float scale, int minimum)
{
    int size = (int)roundf(base * scale);
    return size < minimum ? minimum : size;
}

int main(int argc, char *argv[])
{
#ifdef _WIN32
    const char *model_path = ".\\model\\mnist_model.bin";
#else
    const char *model_path = "./model/mnist_model.bin";
#endif

    if(argc == 2){
        model_path = argv[1];
    }else if(argc != 1){
        fprintf(stderr,
                "Usage: \"%s\" [model_path]\n",
                argv[0]);
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
        fprintf(stderr, "load model failed\n");
        network_free(&net);
        return 1;
    }

    printf("model_layers=%zu\n", net.count);
    fflush(stdout);

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(800, 450, "Digit Recognizer");
    SetWindowMinSize(320, 260);
    SetTargetFPS(60);
    float pixels[28 * 28] = {0};
    float model_pixels[28 * 28] = {0};
    Mat input = {
        .rows = 1,
        .cols = 28 * 28,
        .stride = 28 * 28,
        .es = model_pixels
    };
    Mat scores = mat_alloc(1, 10);
    if (scores.es == NULL) {
        fprintf(stderr, "allocate scores failed\n");
        CloseWindow();
        network_free(&net);
        return 1;
    }

    int prev_col = 0;
    int prev_row = 0;
    int has_previous = 0;
    int prediction = -1;

    while (!WindowShouldClose()) {
        int need_predict = 0;
        int width = GetScreenWidth();
        int height = GetScreenHeight();
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
        int canvas_top = text_top + top_font + gap;
        int footer_height = preview_size + title_font + 2 * gap + bottom_padding;
        int available_width = width - 2 * padding;
        int available_height = height - canvas_top - footer_height;
        int available = available_width < available_height ?
                        available_width : available_height;
        int pixel_size = available / 28;
        if (pixel_size < 1) {
            pixel_size = 1;
        }

        int canvas_size = pixel_size * 28;
        int canvas_x = (width - canvas_size) / 2;
        int canvas_y = canvas_top + (available_height - canvas_size) / 2;
        Vector2 mouse = GetMousePosition();

        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) &&
            mouse.x >= canvas_x && mouse.x < canvas_x + canvas_size &&
            mouse.y >= canvas_y && mouse.y < canvas_y + canvas_size) {
            int col = (int)((mouse.x - canvas_x) * 28 / canvas_size);
            int row = (int)((mouse.y - canvas_y) * 28 / canvas_size);
            if (has_previous) {
                canvas_line(pixels, prev_col, prev_row, col, row);
            } else {
                canvas_brush(pixels, col, row);
            }
            prev_col = col;
            prev_row = row;
            has_previous = 1;
            need_predict = 1;
        } else {
            has_previous = 0;
        }

        if (IsKeyPressed(KEY_C)) {
            for (int i = 0; i < 28 * 28; ++i) {
                pixels[i] = 0.0f;
                model_pixels[i] = 0.0f;
            }
            has_previous = 0;
            prediction = -1;
            need_predict = 0;
        }

        if (need_predict) {
            if (digit_prepare(model_pixels, pixels) &&
                network_forward(scores, &net, input)) {
                prediction = (int)mat_argmax_row(scores, 0);
            } else {
                prediction = -1;
            }
        }

        BeginDrawing();
        ClearBackground(LIGHTGRAY);
        DrawRectangle(canvas_x, canvas_y, canvas_size, canvas_size, BLACK);
        for (int row = 0; row < 28; row++) {
            for (int col = 0; col < 28; col++) {
                float value = pixels[row * 28 + col];
                if (value > 0.0f) {
                    unsigned char gray = (unsigned char)(value * 255.0f);
                    Color color = {gray, gray, gray, 255};
                    DrawRectangle(canvas_x + col * pixel_size,
                                  canvas_y + row * pixel_size,
                                  pixel_size, pixel_size, color);
                }
            }
        }
        if (prediction >= 0) {
            DrawText(TextFormat("Prediction: %d", prediction),
                     padding, text_top, prediction_font, DARKGREEN);
        } else {
            DrawText("Draw a digit | C: clear",
                     padding, text_top, hint_font, DARKGRAY);
        }
        int preview_x = width - preview_size - padding;
        int preview_y = height - preview_size - bottom_padding;
        int title_x = width - padding - MeasureText("Model input", title_font);
        DrawText("Model input", title_x, preview_y - title_font - gap,
                 title_font, DARKGRAY);
        DrawRectangle(preview_x, preview_y, preview_size, preview_size, BLACK);
        for (int row = 0; row < 28; ++row) {
            for (int col = 0; col < 28; ++col) {
                unsigned char gray =
                    (unsigned char)(model_pixels[row * 28 + col] * 255.0f);
                Color color = {gray, gray, gray, 255};
                int x0 = preview_x + col * preview_size / 28;
                int x1 = preview_x + (col + 1) * preview_size / 28;
                int y0 = preview_y + row * preview_size / 28;
                int y1 = preview_y + (row + 1) * preview_size / 28;
                DrawRectangle(x0, y0, x1 - x0, y1 - y0, color);
            }
        }
        DrawText("C: clear", padding, preview_y + gap, helper_font, DARKGRAY);
        DrawText("Draw one digit", padding, preview_y + helper_font + 2 * gap,
                 helper_font, DARKGRAY);
        EndDrawing();
    }

    CloseWindow();
    mat_free(&scores);
    network_free(&net);

    return 0;
}
