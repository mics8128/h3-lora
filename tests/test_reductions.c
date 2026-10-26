/* Fixture-free numeric oracles for multi-SIMD shared-reduction phases. */
#include "h3_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void require(int ok, const char *message) {
    if (!ok) { fprintf(stderr, "FAIL: %s\n", message); exit(1); }
}
static uint16_t bf16(float value) {
    uint32_t bits; memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
static float f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result; memcpy(&result, &bits, sizeof(result)); return result;
}
static void gqa(h3_gpu *gpu, uint32_t sequence) {
    enum { QH = 8, KH = 2, DIM = 128 };
    size_t qn = (size_t)sequence * QH * DIM, kn = (size_t)sequence * KH * DIM;
    uint16_t *q = malloc(qn * 2), *k = malloc(kn * 2), *v = malloc(kn * 2);
    uint16_t *actual = malloc(qn * 2), *first = malloc(qn * 2);
    float *expected = malloc(qn * sizeof(*expected));
    require(q && k && v && actual && first && expected, "GQA oracle allocation");
    for (size_t i = 0; i < qn; i++) q[i] = bf16(1.0f / (float)DIM);
    for (uint32_t row = 0; row < sequence; row++)
        for (uint32_t head = 0; head < KH; head++)
            for (uint32_t d = 0; d < DIM; d++) {
                size_t i = ((size_t)row * KH + head) * DIM + d;
                k[i] = bf16(((float)(row % 37) - 18.0f) * 0.5f + (float)head * 0.125f);
                v[i] = bf16(((float)((row * 13 + d * 7 + head * 3) % 29) - 14.0f) / 16.0f);
            }
    /* Q=1/DIM and each K head constant across dimensions: Q.K is exactly K.
     * Reference softmax uses double, independently of the GPU reduction tree. */
    for (uint32_t row = 0; row < sequence; row++)
        for (uint32_t head = 0; head < QH; head++) {
            uint32_t kv = head / (QH / KH);
            double maximum = -INFINITY, denominator = 0.0;
            for (uint32_t r = 0; r <= row; r++)
                maximum = fmax(maximum, f32(k[((size_t)r * KH + kv) * DIM]));
            for (uint32_t r = 0; r <= row; r++)
                denominator += exp((double)f32(k[((size_t)r * KH + kv) * DIM]) - maximum);
            for (uint32_t d = 0; d < DIM; d++) {
                double sum = 0.0;
                for (uint32_t r = 0; r <= row; r++) {
                    size_t i = ((size_t)r * KH + kv) * DIM;
                    sum += exp((double)f32(k[i]) - maximum) * f32(v[i + d]);
                }
                expected[((size_t)row * QH + head) * DIM + d] = (float)(sum / denominator);
            }
        }
    h3_gpu_tensor *tq = h3_gpu_tensor_from_bf16(gpu, q, qn);
    h3_gpu_tensor *tk = h3_gpu_tensor_from_bf16(gpu, k, kn);
    h3_gpu_tensor *tv = h3_gpu_tensor_from_bf16(gpu, v, kn);
    h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, qn);
    require(tq && tk && tv && out, "GQA GPU allocation");
    for (int repeat = 0; repeat < 24; repeat++) {
        require(h3_gpu_begin(gpu) && h3_gpu_gqa_causal_bf16(
            gpu, out, tq, tk, tv, sequence, QH, KH, DIM, 1.0f) &&
            h3_gpu_submit(gpu) && h3_gpu_tensor_read_bf16(out, actual, qn), "GQA execution");
        float maximum_error = 0.0f;
        for (size_t i = 0; i < qn; i++) {
            float decoded = f32(actual[i]);
            require(isfinite(decoded), "GQA output must be finite");
            require(isfinite(expected[i]), "GQA oracle output must be finite");
            maximum_error = fmaxf(maximum_error, fabsf(decoded - expected[i]));
        }
        if (maximum_error > 0.004f)
            fprintf(stderr, "GQA sequence=%u repeat=%d max-error=%g\n", sequence, repeat, maximum_error);
        require(maximum_error <= 0.004f, "causal GQA disagrees with independent softmax oracle");
        if (!repeat) memcpy(first, actual, qn * 2);
        else require(!memcmp(first, actual, qn * 2), "same Q/K/V must yield identical GQA bytes");
    }
    h3_gpu_tensor_free(tq); h3_gpu_tensor_free(tk); h3_gpu_tensor_free(tv); h3_gpu_tensor_free(out);
    free(q); free(k); free(v); free(actual); free(first); free(expected);
}
static void group_norm(h3_gpu *gpu) {
    enum { DEPTH = 2, HEIGHT = 13, WIDTH = 11, CHANNELS = 64, GROUPS = 8,
           SPATIAL = HEIGHT * WIDTH, COUNT = DEPTH * SPATIAL * CHANNELS };
    float *input = malloc(COUNT * sizeof(float)), *actual = malloc(COUNT * sizeof(float));
    float *first = malloc(COUNT * sizeof(float)), *expected = malloc(COUNT * sizeof(float));
    float weight[CHANNELS], bias[CHANNELS];
    require(input && actual && first && expected, "group norm allocation");
    for (size_t i = 0; i < COUNT; i++) input[i] = 2.0f + ((float)((i * 17) % 61) - 30.0f) / 128.0f;
    for (size_t c = 0; c < CHANNELS; c++) {
        weight[c] = 1.0f + (float)(c % 5) / 16.0f;
        bias[c] = -0.25f + (float)(c % 3) / 32.0f;
    }
    for (size_t t = 0; t < DEPTH; t++)
        for (size_t g = 0; g < GROUPS; g++) {
            double mean = 0.0, variance = 0.0;
            for (size_t s = 0; s < SPATIAL; s++)
                for (size_t c = g * 8; c < (g + 1) * 8; c++)
                    mean += input[(t * SPATIAL + s) * CHANNELS + c];
            mean /= SPATIAL * 8;
            for (size_t s = 0; s < SPATIAL; s++)
                for (size_t c = g * 8; c < (g + 1) * 8; c++) {
                    double x = input[(t * SPATIAL + s) * CHANNELS + c] - mean;
                    variance += x * x;
                }
            variance /= SPATIAL * 8;
            for (size_t s = 0; s < SPATIAL; s++)
                for (size_t c = g * 8; c < (g + 1) * 8; c++) {
                    size_t i = (t * SPATIAL + s) * CHANNELS + c;
                    double x = (input[i] - mean) / sqrt(variance + 1e-6) * weight[c] + bias[c];
                    expected[i] = (float)(x / (1.0 + exp(-x)));
                }
        }
    h3_gpu_tensor *x = h3_gpu_tensor_from_f32(gpu, input, COUNT);
    h3_gpu_tensor *w = h3_gpu_tensor_from_f32(gpu, weight, CHANNELS);
    h3_gpu_tensor *b = h3_gpu_tensor_from_f32(gpu, bias, CHANNELS);
    h3_gpu_tensor *out = h3_gpu_tensor_new_f32(gpu, COUNT);
    require(x && w && b && out, "group norm GPU allocation");
    for (int repeat = 0; repeat < 24; repeat++) {
        require(h3_gpu_begin(gpu) && h3_gpu_vae_encoder_group_norm_silu_f32(
            gpu, out, x, w, b, 1, DEPTH, HEIGHT, WIDTH, CHANNELS, GROUPS, 1e-6f) &&
            h3_gpu_submit(gpu) && h3_gpu_tensor_read_f32(out, actual, COUNT), "group norm execution");
        float maximum_error = 0.0f;
        for (size_t i = 0; i < COUNT; i++) {
            require(isfinite(actual[i]), "group norm output must be finite");
            require(isfinite(expected[i]), "group norm oracle output must be finite");
            maximum_error = fmaxf(maximum_error, fabsf(actual[i] - expected[i]));
        }
        if (maximum_error > 1e-4f) fprintf(stderr, "group norm repeat=%d max-error=%g\n", repeat, maximum_error);
        require(maximum_error <= 1e-4f, "group norm/SiLU disagrees with independent oracle");
        if (!repeat) memcpy(first, actual, COUNT * sizeof(float));
        else require(!memcmp(first, actual, COUNT * sizeof(float)), "same input must yield identical group norm bytes");
    }
    h3_gpu_tensor_free(x); h3_gpu_tensor_free(w); h3_gpu_tensor_free(b); h3_gpu_tensor_free(out);
    free(input); free(actual); free(first); free(expected);
}
int main(void) {
    unsetenv("H3_MPS_GQA");
    char error[512]; h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    require(gpu != NULL, error);
    gqa(gpu, 32); gqa(gpu, 33); gqa(gpu, 168); gqa(gpu, 257);
    group_norm(gpu); h3_gpu_free(gpu);
    puts("Multi-SIMD GQA and encoder group-norm numeric/repeatability checks passed");
    return 0;
}
