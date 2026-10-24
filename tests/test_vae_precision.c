#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void require(int ok, const char *message) {
    if (!ok) {
        fprintf(stderr, "FAIL tests/test_vae_precision.c: %s\n", message);
        exit(1);
    }
}

int main(void) {
    char error[512];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    require(gpu != NULL, error);
    /* BF16 literals: 1, 2, -1, 0.5, 4, -2. Small F32 biases must not
     * be rounded to BF16, nor may the biased result be stored as BF16. */
    const uint16_t input[] = {0x3f80, 0x4000, 0xbf80, 0x3f00, 0x4080, 0xc000};
    const float bias[] = {0.001f, -0.003f, 0.125f};
    const float expected[] = {1.001f, 1.997f, -0.875f,
                              0.501f, 3.997f, -1.875f};
    float actual[6];
    h3_gpu_tensor *in = h3_gpu_tensor_from_bf16(gpu, input, 6);
    h3_gpu_tensor *b = h3_gpu_tensor_from_f32(gpu, bias, 3);
    h3_gpu_tensor *out = h3_gpu_tensor_new_f32(gpu, 6);
    require(in && b && out, "tensor allocation failed");
    require(h3_gpu_begin(gpu) &&
            h3_gpu_bias_bf16_to_f32(gpu, out, in, b, 2, 3) &&
            h3_gpu_submit(gpu) &&
            h3_gpu_tensor_read_f32(out, actual, 6), h3_gpu_error(gpu));
    for (size_t i = 0; i < 6; i++)
        require(fabsf(actual[i] - expected[i]) < 1e-7f,
                "BF16 widening/F32 bias or per-row broadcast is incorrect");
    require(!h3_gpu_bias_bf16_to_f32(gpu, out, b, b, 2, 3),
            "accepted F32 projection input");
    require(!h3_gpu_bias_bf16_to_f32(gpu, in, in, b, 2, 3),
            "accepted BF16 output");
    require(!h3_gpu_bias_bf16_to_f32(gpu, out, in, in, 2, 3),
            "accepted BF16 bias");
    require(!h3_gpu_bias_bf16_to_f32(gpu, out, in, b, 3, 3),
            "accepted undersized input/output");
    require(!h3_gpu_bias_bf16_to_f32(gpu, out, in, b, 1, 4),
            "accepted undersized bias");
    require(!h3_gpu_bias_bf16_to_f32(gpu, out, in, b, 0, 3),
            "accepted zero rows");
    require(!h3_gpu_bias_bf16_to_f32(gpu, out, in, b, UINT32_MAX, 2),
            "accepted overflowing dispatch dimensions");
    h3_gpu_tensor_free(in);
    h3_gpu_tensor_free(b);
    h3_gpu_tensor_free(out);
    h3_gpu_free(gpu);
    puts("ok: BF16 projection widening preserves F32 bias and validates boundaries");
    return 0;
}
