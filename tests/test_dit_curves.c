#include "h3_dit_schedule.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void check(int ok, const char *message) {
    if (!ok) { fprintf(stderr, "curve test: %s\n", message); exit(1); }
}

static void fixture_data(const h3_st_header *header, const char *prefix,
                         const char *name, h3_dtype dtype, void *data, size_t bytes) {
    char error[512], key[128];
    snprintf(key, sizeof(key), "%s.%s", prefix, name);
    const h3_st_tensor *tensor = h3_st_find(header, key);
    check(tensor && tensor->dtype == dtype &&
          tensor->data_end - tensor->data_begin == bytes, "invalid Torch fixture tensor");
    check(h3_st_read_data(header, tensor, data, bytes, error, sizeof(error)), error);
}

static void mixed_final_fixture(h3_gpu *gpu, const char *path,
                                 const char *prefix, uint32_t width) {
    enum { MAX_WIDTH = 5376 };
    check(width <= MAX_WIDTH, "unsupported fixture width");
    h3_st_header header;
    char error[512];
    check(h3_st_read_header(path, &header, error, sizeof(error)), error);
    uint16_t input[4 * MAX_WIDTH], weight[MAX_WIDTH];
    float modulation[8 * MAX_WIDTH], expected[3 * MAX_WIDTH], got[3 * MAX_WIDTH];
    uint32_t row_map[3];
    fixture_data(&header, prefix, "input", H3_DTYPE_BF16, input, 4 * width * sizeof(*input));
    fixture_data(&header, prefix, "weight", H3_DTYPE_BF16, weight, width * sizeof(*weight));
    fixture_data(&header, prefix, "modulation", H3_DTYPE_F32, modulation, 8 * width * sizeof(*modulation));
    fixture_data(&header, prefix, "expected", H3_DTYPE_F32, expected, 3 * width * sizeof(*expected));
    fixture_data(&header, prefix, "row_map", H3_DTYPE_I32, row_map, sizeof(row_map));
    h3_st_free_header(&header);
    h3_gpu_tensor *x = h3_gpu_tensor_from_bf16(gpu, input, 4 * width);
    h3_gpu_tensor *w = h3_gpu_tensor_from_bf16(gpu, weight, width);
    h3_gpu_tensor *m = h3_gpu_tensor_from_f32(gpu, modulation, 8 * width);
    h3_gpu_tensor *map = h3_gpu_tensor_from_u32(gpu, row_map, 3);
    h3_gpu_tensor *out = h3_gpu_tensor_new_f32(gpu, 3 * width);
    check(x && w && m && map && out, "mixed fixture allocation failed");
    check(h3_gpu_begin(gpu) && h3_gpu_adaln_bf16_f32_offset(
        gpu, out, x, width, w, m, map, 3, width, 2, 0, 1, 1e-5f) &&
        h3_gpu_submit(gpu), h3_gpu_error(gpu));
    check(h3_gpu_tensor_read_f32(out, got, 3 * width), "mixed fixture readback failed");
    for (unsigned index = 0; index < 3 * width; index++)
        check(isfinite(got[index]) && fabsf(got[index] - expected[index]) < 2e-6f,
              "mixed final modulation differs from actual Torch BF16 RMSNorm/F32 modulation");
    check(!h3_gpu_adaln_bf16_f32_offset(gpu, out, x, SIZE_MAX, w, m, map,
        3, width, 2, 0, 1, 1e-5f), "overflowing offset accepted");
    check(!h3_gpu_adaln_bf16_f32_offset(gpu, out, x, width, w, x, map,
        3, width, 2, 0, 1, 1e-5f), "BF16 modulation accepted by F32 curve kernel");
    check(!h3_gpu_adaln_bf16_f32_offset(gpu, out, x, width, w, m, map,
        3, width, 2, 2, 1, 1e-5f), "invalid modulation slot accepted");
    h3_gpu_tensor_free(x); h3_gpu_tensor_free(w);
    h3_gpu_tensor_free(m); h3_gpu_tensor_free(map); h3_gpu_tensor_free(out);
}

int main(int argc, char **argv) {
    check(argc == 2, "usage: test_dit_curves TORCH_FIXTURE.safetensors");
    float table[1025 * 8];
    for (unsigned row = 0; row < 1025; row++)
        for (unsigned column = 0; column < 8; column++)
            table[row * 8 + column] = (float)(row * row) / 1048576.0f + (float)column / 16.0f;
    const float times[] = {0.0f, 0.999f, 1.0f, 0.12345f, -0.5f, 1.5f};
    /* Independent torch.lerp oracle, PyTorch 2.14.1 CPU float32. */
    const float expected_base[] = {0.0f, 0.9980010390281677f, 1.0f,
                                  0.015240134671330452f, 0.0f, 1.0f};
    float input[6 * 8], weight[3 * 8], bias[3] = {-0.25f, 0.5f, 0.125f};
    for (unsigned row = 0; row < 6; row++) {
        check(h3_dit_curve_interpolate(table, 1025, 8, times[row], input + row * 8), "valid interpolation rejected");
        float base = expected_base[row];
        for (unsigned column = 0; column < 8; column++) {
            float want = base + (float)column / 16.0f;
            check(fabsf(input[row * 8 + column] - want) < 2e-7f, "curve Torch oracle mismatch");
        }
    }
    check(!h3_dit_curve_interpolate(table, 1024, 8, 0.5f, input), "unsupported grid accepted");
    check(!h3_dit_curve_interpolate(table, 1025, 7, 0.5f, input), "unsupported rank accepted");
    check(!h3_dit_curve_interpolate(table, 1025, 8, NAN, input), "NaN accepted");
    for (unsigned i = 0; i < 24; i++) weight[i] = ((float)i - 11.0f) / 32.0f;
    char error[512];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    check(gpu != NULL, error);
    h3_gpu_tensor *x = h3_gpu_tensor_from_f32(gpu, input, 48);
    h3_gpu_tensor *w = h3_gpu_tensor_from_f32(gpu, weight, 24);
    h3_gpu_tensor *b = h3_gpu_tensor_from_f32(gpu, bias, 3);
    h3_gpu_tensor *y = h3_gpu_tensor_new_f32(gpu, 18);
    check(x && w && b && y, "allocation failed");
    check(h3_gpu_begin(gpu) && h3_gpu_linear_f32(gpu, y, x, w, b, 6, 8, 3) && h3_gpu_submit(gpu), h3_gpu_error(gpu));
    float got[18];
    check(h3_gpu_tensor_read_f32(y, got, 18), "readback failed");
    for (unsigned row = 0; row < 6; row++)
        for (unsigned out = 0; out < 3; out++) {
            double want = bias[out];
            for (unsigned k = 0; k < 8; k++) want += (double)input[row * 8 + k] * weight[out * 8 + k];
            check(fabs(got[row * 3 + out] - want) < 2e-6, "rank8 GPU projection differs from direct F32 linear (no SiLU)");
        }
    h3_gpu_tensor_free(x); h3_gpu_tensor_free(w);
    h3_gpu_tensor_free(b); h3_gpu_tensor_free(y);
    mixed_final_fixture(gpu, argv[1], "small", 16);
    mixed_final_fixture(gpu, argv[1], "production", 5376);
    h3_gpu_free(gpu);
    puts("curve interpolation, rank8 projection, and Torch final precision fixture passed");
    return 0;
}
