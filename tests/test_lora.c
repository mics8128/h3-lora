#include "h3_lora.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void require(int ok, const char *why) {
    if (!ok) {
        fprintf(stderr, "FAIL test_lora: %s\n", why);
        exit(1);
    }
}
static void compose(h3_lora_target *t, size_t first, size_t rows, void *values) {
    char e[512];
    size_t n = h3_lora_scratch_bytes(t, rows);
    void *s = malloc(n);
    require(s != NULL, "scratch");
    require(h3_lora_merge_rows(t, first, rows, values, s, n, e, sizeof(e)), e);
    free(s);
}
static void numeric(void) {
    {
        float aa[] = {1, 2}, ba[] = {3, 4}, ai[] = {1, 0, 0, 1}, bi[] = {2, 0, 0, -2};
        h3_lora_factor pair[] = {{aa, ba, H3_DTYPE_F32, H3_DTYPE_F32, 1, 1},
                                 {ai, bi, H3_DTYPE_F32, H3_DTYPE_F32, 2, -1}};
        const float answers[][4] = {
            {1, 2, 3, 4}, {4, 8, 7, 12}, {-1, 2, 3, 6}, {2, 8, 7, 14}};
        for (int mask = 0; mask < 4; mask++) {
            float w[] = {1, 2, 3, 4};
            h3_lora_factor selected[2];
            size_t count = 0;
            for (int j = 0; j < 2; j++)
                if (mask & (1 << j))
                    selected[count++] = pair[j];
            h3_lora_target exact = {.name = "exact",
                                    .rows = 2,
                                    .columns = 2,
                                    .dtype = H3_DTYPE_F32,
                                    .factors = selected,
                                    .factor_count = count};
            compose(&exact, 0, 2, w);
            require(!memcmp(w, answers[mask], sizeof(w)),
                    "specified exact additive matrix");
        }
    }
    float a[] = {1, 2, 3, 4, 5, 6}, b[] = {2, 3, 4, 5},
          base[] = {10, 20, 30, 40, 50, 60};
    h3_lora_factor factors[] = {{a, b, H3_DTYPE_F32, H3_DTYPE_F32, 2, 0.5f},
                                {a, b, H3_DTYPE_F32, H3_DTYPE_F32, 2, -0.25f}};
    h3_lora_target t = {.name = "tiny",
                        .rows = 2,
                        .columns = 3,
                        .dtype = H3_DTYPE_F32,
                        .factors = factors,
                        .factor_count = 2};
    compose(&t, 0, 2, base);
    const float expected[] = {13.5f, 24.75f, 36, 46, 58.25f, 70.5f};
    for (int i = 0; i < 6; i++)
        require(base[i] == expected[i], "signed additive F32 composition");
    uint16_t ab[] = {0x3f80, 0x4000, 0x4040}, bb[] = {0x3b80},
             wb[] = {0x3f80, 0x3f80, 0x3f80};
    h3_lora_factor f = {ab, bb, H3_DTYPE_BF16, H3_DTYPE_BF16, 1, 1};
    t = (h3_lora_target){.name = "rounding",
                         .rows = 1,
                         .columns = 3,
                         .dtype = H3_DTYPE_BF16,
                         .factors = &f,
                         .factor_count = 1};
    compose(&t, 0, 1, wb);
    require(wb[0] == 0x3f80 && wb[1] == 0x3f81 && wb[2] == 0x3f82, "BF16 ties to even");
    h3_lora_factor twice[] = {f, f};
    t.factors = twice;
    t.factor_count = 2;
    wb[0] = wb[1] = wb[2] = 0x3f80;
    compose(&t, 0, 1, wb);
    require(wb[0] == 0x3f81 && wb[1] == 0x3f82 && wb[2] == 0x3f83,
            "single BF16 rounding after all adapters");
    float qa[] = {1}, qb[12], out[12] = {0};
    for (int i = 0; i < 12; i++)
        qb[i] = (float)(i + 1);
    f = (h3_lora_factor){qa, qb, H3_DTYPE_F32, H3_DTYPE_F32, 1, 1};
    t = (h3_lora_target){.name = "qkv",
                         .rows = 12,
                         .columns = 1,
                         .dtype = H3_DTYPE_F32,
                         .heads = 2,
                         .head_dim = 2,
                         .factors = &f,
                         .factor_count = 1};
    compose(&t, 0, 12, out);
    const float qexpected[] = {1, 2, 5, 6, 9, 10, 3, 4, 7, 8, 11, 12};
    require(!memcmp(out, qexpected, sizeof(out)), "generic QKV B row permutation");
    float tail[3] = {0};
    compose(&t, 5, 3, tail);
    require(tail[0] == 10 && tail[1] == 3 && tail[2] == 4, "nonaligned QKV row tile");
    /* Rank >64 crosses the fixed rank panel boundary. */
    float ra[65], rb[65];
    for (int i = 0; i < 65; i++)
        ra[i] = rb[i] = 1;
    f = (h3_lora_factor){ra, rb, H3_DTYPE_F32, H3_DTYPE_F32, 65, 1};
    t = (h3_lora_target){.name = "panels",
                         .rows = 1,
                         .columns = 1,
                         .dtype = H3_DTYPE_F32,
                         .factors = &f,
                         .factor_count = 1};
    float v = 2;
    compose(&t, 0, 1, &v);
    require(v == 67, "rank panel accumulation");
    {
        float bigb[130 * 65], rows[130];
        for (size_t i = 0; i < 130 * 65; i++)
            bigb[i] = 1;
        for (int i = 0; i < 130; i++)
            rows[i] = 2;
        f.b = bigb;
        t.rows = 130;
        compose(&t, 0, 128, rows);
        compose(&t, 128, 2, rows + 128);
        for (int i = 0; i < 130; i++)
            require(rows[i] == 67, "row and rank panel tail composition");
        f.b = rb;
        t.rows = 1;
    }
    {
        uint16_t mixedb[] = {0x4000};
        float mixeda[] = {0x1p-16f};
        h3_lora_factor mixed = {mixeda, mixedb, H3_DTYPE_F32, H3_DTYPE_BF16, 1, 1};
        h3_lora_target mt = {.name = "mixed",
                             .rows = 1,
                             .columns = 1,
                             .dtype = H3_DTYPE_F32,
                             .factors = &mixed,
                             .factor_count = 1};
        float result = 1;
        compose(&mt, 0, 1, &result);
        require(result == 1 + 0x1p-15f,
                "mixed factor widening and F32 destination precision");
    }
    {
        unsigned char unaligned[1 + 3 * sizeof(float)];
        const float input[] = {1, 2, 3};
        memcpy(unaligned + 1, input, sizeof(input));
        float up[] = {2}, output[] = {4, 5, 6};
        h3_lora_factor factor = {unaligned + 1, up, H3_DTYPE_F32, H3_DTYPE_F32, 1, 1};
        h3_lora_target target = {.name = "unaligned",
                                 .rows = 1,
                                 .columns = 3,
                                 .dtype = H3_DTYPE_F32,
                                 .factors = &factor,
                                 .factor_count = 1};
        compose(&target, 0, 1, output);
        require(output[0] == 6 && output[1] == 9 && output[2] == 12,
                "unaligned snapshot F32 panels use aligned scratch");
    }
    {
        uint16_t half[] = {0x3c00, 0xc000, 0x0001, 0x03ff, 0x8000, 0};
        float up[] = {1}, output[6] = {0};
        h3_lora_factor factor = {half, up, H3_DTYPE_F16, H3_DTYPE_F32, 1, 1};
        h3_lora_target target = {.name = "half", .rows = 1, .columns = 6,
                                 .dtype = H3_DTYPE_F32, .factors = &factor,
                                 .factor_count = 1};
        compose(&target, 0, 1, output);
        require(output[0] == 1 && output[1] == -2 && output[2] == 0x1p-24f &&
                    output[3] == 1023 * 0x1p-24f && output[4] == 0 && output[5] == 0,
                "IEEE half normal, subnormal and signed zero mixed factors");
        uint16_t bfup[] = {0x4000};
        factor.b = bfup;
        factor.b_dtype = H3_DTYPE_BF16;
        memset(output, 0, sizeof(output));
        compose(&target, 0, 1, output);
        require(output[0] == 2 && output[2] == 0x1p-23f, "mixed F16/BF16 factors");
        char error[128];
        size_t bytes = h3_lora_scratch_bytes(&target, 1);
        void *scratch = malloc(bytes);
        require(scratch != NULL, "half scratch");
        target.dtype = H3_DTYPE_F16;
        require(!h3_lora_merge_rows(&target, 0, 1, output, scratch, bytes,
                                    error, sizeof(error)), "F16 destination rejected");
        target.dtype = H3_DTYPE_F32;
        half[0] = 0x7c00;
        require(!h3_lora_merge_rows(&target, 0, 1, output, scratch, bytes,
                                    error, sizeof(error)), "half infinity rejected");
        half[0] = 0x7e00;
        require(!h3_lora_merge_rows(&target, 0, 1, output, scratch, bytes,
                                    error, sizeof(error)), "half NaN rejected");
        free(scratch);
    }
    char e[128];
    size_t bytes = h3_lora_scratch_bytes(&t, 1);
    void *s = malloc(bytes);
    ra[0] = INFINITY;
    v = 0;
    require(!h3_lora_merge_rows(&t, 0, 1, &v, s, bytes, e, sizeof(e)),
            "nonfinite merge rejection");
    free(s);
    t.columns = 14336;
    require(h3_lora_scratch_bytes(&t, 128) + 128 * 14336 * 4 < 18u * 1024 * 1024,
            "bounded workspace");
}
static void file(const char *path, const char *json, const void *data, size_t size) {
    FILE *f = fopen(path, "wb");
    require(f != NULL, "fixture open");
    size_t n = strlen(json);
    unsigned char prefix[8];
    for (int i = 0; i < 8; i++)
        prefix[i] = (unsigned char)((uint64_t)n >> (8 * i));
    require(fwrite(prefix, 1, 8, f) == 8 && fwrite(json, 1, n, f) == n &&
                fwrite(data, 1, size, f) == size,
            "fixture write");
    require(!fclose(f), "fixture close");
}
static void adapter(const char *path, const char *stem, int alpha, float av, float a0,
                    float b0) {
    char json[2048];
    size_t an = 5376 * 4, bn = 32 * 4;
    float *data = calloc(5376 + 32 + 1, 4);
    require(data != NULL, "adapter fixture");
    for (int i = 0; i < 5376; i++)
        data[i] = a0;
    for (int i = 0; i < 32; i++)
        data[5376 + i] = b0;
    data[5376 + 32] = av;
    snprintf(json, sizeof(json),
             "{\"%s.lora_A.weight\":{\"dtype\":\"F32\",\"shape\":[1,5376],\"data_"
             "offsets\":[0,%zu]},\"%s.lora_B.weight\":{\"dtype\":\"F32\",\"shape\":[32,"
             "1],\"data_offsets\":[%zu,%zu]}%s",
             stem, an, stem, an, an + bn, alpha ? "," : "}");
    if (alpha) {
        size_t used = strlen(json);
        snprintf(
            json + used, sizeof(json) - used,
            "\"%s.alpha\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[%zu,%zu]}}",
            stem, an + bn, an + bn + 4);
    }
    file(path, json, data, an + bn + (alpha ? 4 : 0));
    free(data);
}
static void metadata_adapter(const char *path, const char *metadata, int alpha,
                             uint16_t alpha_bits) {
    size_t count = 5376 + 32;
    unsigned char *data = calloc(count * 4 + 2, 1);
    require(data != NULL, "metadata fixture");
    float one = 1;
    for (size_t i = 0; i < count; i++) memcpy(data + 4 * i, &one, 4);
    memcpy(data + count * 4, &alpha_bits, 2);
    char json[4096], suffix[256] = "";
    if (alpha)
        snprintf(suffix, sizeof(suffix), ",\"final_layer.audio_out.alpha\":{\"dtype\":\"F16\",\"shape\":[],\"data_offsets\":[21632,21634]}");
    snprintf(json, sizeof(json),
             "{\"__metadata__\":{%s},\"final_layer.audio_out.lora_A.weight\":{\"dtype\":\"F32\",\"shape\":[1,5376],\"data_offsets\":[0,21504]},\"final_layer.audio_out.lora_B.weight\":{\"dtype\":\"F32\",\"shape\":[32,1],\"data_offsets\":[21504,21632]}%s}", metadata, suffix);
    file(path, json, data, count * 4 + (alpha ? 2 : 0));
    free(data);
}
static void parser_validation(void) {
    static const char *bad[] = {
        "{\"x\":{\"dtype\":\"F32\",\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,"
        "4]}}",
        "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"x\":{"
        "\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}}",
        "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"y\":{"
        "\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}"};
    for (size_t i = 0; i < sizeof(bad) / sizeof(*bad); i++) {
        size_t n = strlen(bad[i]);
        unsigned char *p = calloc(n + 16, 1);
        for (int j = 0; j < 8; j++)
            p[j] = (unsigned char)((uint64_t)n >> (8 * j));
        memcpy(p + 8, bad[i], n);
        h3_st_header h;
        char e[128];
        require(!h3_st_read_memory(p, n + 16, "invalid", &h, e, sizeof(e)),
                "strict parser malformed fixture");
        free(p);
    }
}
static void quantization(h3_gpu *gpu) {
    if (!h3_gpu_is_m5(gpu)) {
        puts("SKIP int8 fixture: M5 tensor operations required");
        return;
    }
    enum { DIM = 128, COUNT = DIM * DIM };
    uint16_t composed[COUNT], expected[COUNT], input[COUNT], oa[COUNT], ob[COUNT];
    float a[DIM], b[DIM];
    for (int i = 0; i < DIM; i++) {
        a[i] = 1;
        b[i] = 2;
    }
    for (int i = 0; i < COUNT; i++) {
        composed[i] = 0x3f80;
        expected[i] = 0x4040;
        input[i] = 0x3f80;
    }
    h3_lora_factor f = {a, b, H3_DTYPE_F32, H3_DTYPE_F32, 1, 1};
    h3_lora_target t = {.name = "int8",
                        .rows = DIM,
                        .columns = DIM,
                        .dtype = H3_DTYPE_BF16,
                        .factors = &f,
                        .factor_count = 1};
    compose(&t, 0, DIM, composed);
    h3_gpu_tensor *wa = h3_gpu_tensor_from_bf16(gpu, composed, COUNT),
                  *wb = h3_gpu_tensor_from_bf16(gpu, expected, COUNT);
    h3_gpu_tensor *qa = h3_gpu_tensor_new_i8(gpu, COUNT),
                  *qb = h3_gpu_tensor_new_i8(gpu, COUNT);
    h3_gpu_tensor *sa = h3_gpu_tensor_new_f32(gpu, DIM),
                  *sb = h3_gpu_tensor_new_f32(gpu, DIM);
    h3_gpu_tensor *x = h3_gpu_tensor_from_bf16(gpu, input, COUNT),
                  *qx = h3_gpu_tensor_new_i8(gpu, COUNT),
                  *sx = h3_gpu_tensor_new_f32(gpu, DIM);
    h3_gpu_tensor *ya = h3_gpu_tensor_new_bf16(gpu, COUNT),
                  *yb = h3_gpu_tensor_new_bf16(gpu, COUNT);
    require(wa && wb && qa && qb && sa && sb && x && qx && sx && ya && yb,
            "int8 tensors");
    require(h3_gpu_begin(gpu), "int8 command begin");
    require(h3_gpu_quantize_weight_int8(gpu, qa, sa, wa, DIM, DIM) &&
                h3_gpu_quantize_weight_int8(gpu, qb, sb, wb, DIM, DIM),
            "production quantization");
    require(h3_gpu_linear_int8_bf16(gpu, ya, qx, sx, x, qa, sa, DIM, DIM, DIM, 0) &&
                h3_gpu_linear_int8_bf16(gpu, yb, qx, sx, x, qb, sb, DIM, DIM, DIM, 0),
            "quantized projections");
    require(h3_gpu_submit(gpu), "int8 submission");
    float scales_a[DIM], scales_b[DIM];
    require(h3_gpu_tensor_read_f32(sa, scales_a, DIM) &&
                h3_gpu_tensor_read_f32(sb, scales_b, DIM),
            "quantization scale readback");
    require(!memcmp(scales_a, scales_b, sizeof(scales_a)),
            "adapted scales equal independent expected scales");
    require(h3_gpu_tensor_read_bf16(ya, oa, COUNT) &&
                h3_gpu_tensor_read_bf16(yb, ob, COUNT),
            "int8 projection readback");
    require(!memcmp(oa, ob, sizeof(oa)),
            "adapted quantization equals independent expected projection");
    for (int i = 0; i < COUNT; i++)
        require(oa[i] == 0x43c0, "quantized scalar projection oracle 384");
    h3_gpu_tensor *all[] = {wa, wb, qa, qb, sa, sb, x, qx, sx, ya, yb};
    for (size_t i = 0; i < sizeof(all) / sizeof(*all); i++)
        h3_gpu_tensor_free(all[i]);
}


static void parallel_loader(h3_gpu *gpu) {
    enum { ROW_COUNT = 1542, COLUMNS = 8191, FACTORS = 3 };
    char path[] = "/tmp/h3-lora-rows-XXXXXX", error[512];
    int fd = mkstemp(path);
    require(fd >= 0, "parallel loader fixture");
    close(fd);
    h3_lora_factor factors[FACTORS];
    void *a[FACTORS], *b[FACTORS];
    h3_dtype types[] = {H3_DTYPE_BF16, H3_DTYPE_F16, H3_DTYPE_F32};
    for (size_t f = 0; f < FACTORS; f++) {
        size_t rank = 65 + f, item = h3_dtype_size(types[f]);
        a[f] = malloc(rank * COLUMNS * item + 1);
        b[f] = malloc(rank * ROW_COUNT * item + 1);
        require(a[f] && b[f], "parallel factor allocation");
        void *arrays[] = {a[f], b[f]};
        size_t counts[] = {rank * COLUMNS, rank * ROW_COUNT};
        for (size_t part = 0; part < 2; part++)
            for (size_t i = 0; i < counts[part]; i++) {
                if (types[f] == H3_DTYPE_F32) {
                    float v = ((float)((i * 17 + f + part) % 23) - 11.0f) / 257.0f;
                    memcpy((char *)arrays[part] + 1 + i * item, &v, item);
                } else {
                    uint16_t v = (uint16_t)((types[f] == H3_DTYPE_F16 ? 0x2800 : 0x3c00)
                        + (i * 37 + f + part) % 512);
                    if (i % 3 == 0) v |= 0x8000;
                    memcpy((char *)arrays[part] + 1 + i * item, &v, item);
                }
            }
        factors[f] = (h3_lora_factor){(char *)a[f] + 1, (char *)b[f] + 1,
            types[f], types[f], rank, f == 1 ? -0.5f : 0.25f};
    }
    for (int format = 0; format < 2; format++) {
        h3_dtype dtype = format ? H3_DTYPE_F32 : H3_DTYPE_BF16;
        size_t elements = (size_t)ROW_COUNT * COLUMNS, item = h3_dtype_size(dtype);
        void *expected = malloc(elements * item), *actual = malloc(elements * item);
        require(expected && actual, "parallel merged allocation");
        for (size_t i = 0; i < elements; i++) {
            if (format) {
                float v = 1.0f + (float)(i % 7) / 128.0f;
                memcpy((char *)expected + i * item, &v, item);
            } else {
                uint16_t v = (uint16_t)(0x3f80 + i % 7);
                memcpy((char *)expected + i * item, &v, item);
            }
        }
        FILE *stream = fopen(path, "wb");
        require(stream != NULL, "parallel base open");
        require(fwrite(expected, item, elements, stream) == elements, "parallel base write");
        require(!fclose(stream), "parallel base close");
        h3_lora_target target = {.name = "parallel-qkv", .rows = ROW_COUNT,
            .columns = COLUMNS, .dtype = dtype, .heads = 257, .head_dim = 2,
            .factors = factors, .factor_count = FACTORS};
        for (size_t row = 0; row < ROW_COUNT; row += 128) {
            size_t count = ROW_COUNT - row;
            if (count > 128) count = 128;
            compose(&target, row, count, (char *)expected + row * COLUMNS * item);
        }
        h3_st_header header = {.path = path};
        h3_st_tensor base = {.name = target.name, .dtype = dtype, .ndim = 2,
                            .shape = {ROW_COUNT, COLUMNS}};
        h3_gpu_tensor *loaded = h3_lora_load_target(&target, gpu, &header, &base,
                                                   error, sizeof(error));
        require(loaded != NULL, error);
        require(format ? h3_gpu_tensor_read_f32(loaded, actual, elements) :
                         h3_gpu_tensor_read_bf16(loaded, actual, elements),
                "parallel merged readback");
        require(!memcmp(expected, actual, elements * item),
                "parallel rows match serial rank-panel arithmetic for all bytes");
        h3_gpu_tensor_free(loaded);
        stream = fopen(path, "wb");
        require(stream && !fclose(stream), "truncate parallel source");
        loaded = h3_lora_load_target(&target, gpu, &header, &base, error, sizeof(error));
        require(loaded == NULL, "parallel read failure joins before returning no tensor");
        free(expected); free(actual);
    }
    for (size_t f = 0; f < FACTORS; f++) { free(a[f]); free(b[f]); }
    require(!unlink(path), "parallel loader cleanup");
}

static void stack_and_metal(void) {
    char dir[] = "/tmp/h3-lora-test-XXXXXX";
    require(mkdtemp(dir) != NULL, "temporary directory");
    char bp[512], ap[512], cp[512];
    snprintf(bp, sizeof(bp), "%s/base.safetensors", dir);
    snprintf(ap, sizeof(ap), "%s/adapter.bin", dir);
    snprintf(cp, sizeof(cp), "%s/second.bin", dir);
    size_t elements = 32 * 5376;
    float *base = calloc(elements, 4);
    require(base != NULL, "base fixture");
    for (size_t i = 0; i < elements; i++)
        base[i] = 1;
    file(bp,
         "{\"final_layer.audio_out.weight\":{\"dtype\":\"F32\",\"shape\":[32,5376],"
         "\"data_offsets\":[0,688128]}}",
         base, elements * 4);
    adapter(ap, "diffusion_model.final_layer.audio_out", 1, 0.5f, 2, 3);
    adapter(cp, "final_layer.audio_out", 0, 0, 1, 2);
    char e[512];
    h3_weight_store *store = h3_weight_store_open(dir, e, sizeof(e));
    require(store != NULL, e);
    h3_lora_spec zero = {"/not/present", -0.0f};
    {
        const char *ordinary = "\"ss_network_module\":\"networks.lora\",\"ss_network_alpha\":\"0.25\",\"ss_network_dim\":\"99\"";
        h3_lora_spec fixture = {ap, 1};
        for (int explicit = 0; explicit < 2; explicit++) {
            metadata_adapter(ap, ordinary, explicit, 0x3800);
            h3_lora_stack *ms = h3_lora_stack_load(&fixture, 1, store, e, sizeof(e));
            require(ms != NULL, e);
            float result[5376] = {0};
            compose((h3_lora_target *)h3_lora_find_target(ms, "final_layer.audio_out.weight"), 0, 1, result);
            require(result[17] == (explicit ? 0.5f : 0.25f), "metadata alpha applied; F16 explicit alpha wins; dim ignored");
            h3_lora_stack_free(ms);
        }
        const char *bad[] = {
            "\"ss_network_module\":\"networks.lora\",\"ss_network_alpha\":\"oops\"",
            "\"ss_network_module\":\"networks.lora\",\"ss_network_alpha\":\"nan\"",
            "\"ss_network_alpha\":\"0.25\"",
            "\"ss_network_alpha\":\"Dynamic\"",
            "\"ss_network_args\":\"{}\"",
            "\"rslora\":\"true\"",
            "\"dora\":\"true\""};
        for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
            metadata_adapter(ap, bad[i], 0, 0);
            require(!h3_lora_stack_load(&fixture, 1, store, e, sizeof(e)), "unsupported metadata fails closed");
        }
        metadata_adapter(ap, ordinary, 1, 0x7c00);
        require(!h3_lora_stack_load(&fixture, 1, store, e, sizeof(e)), "nonfinite F16 alpha rejected");
        const char *normalized = "\"ss_network_alpha\":\"Dynamic\",\"alpha_normalized\":\"true\",\"alpha_normalization\":\"lora_up := lora_up * (alpha / rank); alpha tensors removed\"";
        const char *baked = "\"ss_network_alpha\":\"Dynamic\",\"converted_layout\":\"comfyui_minimax_h3\",\"baked_scale\":\"0.125\",\"conversion\":\"qkv block-diag fused (per-projection ranks); mlp.fc1 swiglu halves swapped; alpha dropped (alpha==rank, scale 1.0); effective scale 0.125 baked into lora_B (flat multiplier, NOT alpha/rank)\"";
        const char *accepted[] = {normalized, baked, "\"ss_network_alpha\":\"Dynamic\""};
        for (int i = 0; i < 3; i++) {
            metadata_adapter(ap, accepted[i], i == 2, 0x3800);
            h3_lora_stack *ms = h3_lora_stack_load(&fixture, 1, store, e, sizeof(e));
            require(ms != NULL, e);
            float result[5376] = {0};
            compose((h3_lora_target *)h3_lora_find_target(ms, "final_layer.audio_out.weight"), 0, 1, result);
            require(result[0] == (i == 2 ? 0.5f : 1), "baked scaling not reapplied; Dynamic uses explicit alpha");
            h3_lora_stack_free(ms);
        }
        metadata_adapter(ap, "\"baked_scale\":\"0.125\",\"conversion\":\"unknown\"", 0, 0);
        require(!h3_lora_stack_load(&fixture, 1, store, e, sizeof(e)), "unknown baked semantics rejected");
        adapter(ap, "diffusion_model.final_layer.audio_out", 1, 0.5f, 2, 3);
    }
    require(!h3_lora_stack_load(&zero, 1, store, e, sizeof(e)) && !e[0],
            "zero strength never opens path");
    h3_lora_spec specs[] = {{ap, 1}, {cp, -0.25f}, {ap, 0.5f}};
    h3_lora_stack *s = h3_lora_stack_load(specs, 3, store, e, sizeof(e));
    require(s != NULL, e);
    h3_lora_spec reversed[] = {specs[2], specs[1], specs[0]};
    h3_lora_stack *r = h3_lora_stack_load(reversed, 3, store, e, sizeof(e));
    require(r != NULL, e);
    require(!strcmp(h3_lora_stack_identity(s), h3_lora_stack_identity(r)),
            "canonical multiset identity");
    const h3_lora_target *t = h3_lora_find_target(s, "final_layer.audio_out.weight"),
                         *rt = h3_lora_find_target(r, "final_layer.audio_out.weight");
    require(t && rt, "semantic wrapper mapping");
    float *row = malloc(5376 * 4), *other = malloc(5376 * 4);
    for (int i = 0; i < 5376; i++)
        row[i] = other[i] = 1;
    compose((h3_lora_target *)t, 0, 1, row);
    compose((h3_lora_target *)rt, 0, 1, other);
    require(!memcmp(row, other, 5376 * 4), "canonical composition bytes");
    require(row[17] == 5, "alpha/rank and signed duplicate scales");
    {
        h3_lora_spec ca[] = {{ap, 1e8f}, {ap, 1}, {ap, -1e8f}},
                     cb[] = {ca[2], ca[1], ca[0]};
        h3_lora_stack *as = h3_lora_stack_load(ca, 3, store, e, sizeof(e));
        require(as != NULL, e);
        h3_lora_stack *bs = h3_lora_stack_load(cb, 3, store, e, sizeof(e));
        require(bs != NULL, e);
        for (int i = 0; i < 5376; i++)
            row[i] = other[i] = 1;
        compose((h3_lora_target *)h3_lora_find_target(as, t->name), 0, 1, row);
        compose((h3_lora_target *)h3_lora_find_target(bs, t->name), 0, 1, other);
        require(!memcmp(row, other, 5376 * 4),
                "cancellation-sensitive canonical order");
        h3_lora_stack_free(as);
        h3_lora_stack_free(bs);
    }
    /* Destroy source artifacts after snapshot, then compose a different tile. */
    require(!unlink(ap) && !unlink(cp), "remove adapter sources");
    for (int i = 0; i < 5376; i++)
        other[i] = 1;
    compose((h3_lora_target *)t, 17, 1, other);
    require(other[17] == 5, "immutable snapshot survives source deletion");
    require(h3_weight_store_bind_loras(store, s, e, sizeof(e)), e);
    require(!h3_weight_store_bind_loras(store, s, e, sizeof(e)), "rebind rejected");
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", e, sizeof(e));
    require(gpu != NULL, e);
    uint64_t shape[] = {32, 5376};
    h3_gpu_tensor *w = h3_weight_load_f32(store, gpu, t->name, 2, shape, e, sizeof(e));
    require(w != NULL, e);
    float *readback = malloc(elements * 4);
    require(h3_gpu_tensor_read_f32(w, readback, elements),
            "Metal adapted weight readback");
    for (size_t i = 0; i < elements; i++)
        require(readback[i] == 5, "Metal weight store composed payload");
    /* Execute actual projection against the adapted weight, not just upload. */
    float *input = calloc(5376, 4);
    require(input != NULL, "projection input");
    input[0] = 1;
    input[17] = 2;
    h3_gpu_tensor *x = h3_gpu_tensor_from_f32(gpu, input, 5376),
                  *y = h3_gpu_tensor_new_f32(gpu, 32);
    require(h3_gpu_begin(gpu), "projection command begin");
    require(x && y, "projection tensors");
    require(h3_gpu_linear_f32(gpu, y, x, w, NULL, 1, 5376, 32), "adapted projection");
    require(h3_gpu_submit(gpu), "projection submission");
    float projected[32];
    require(h3_gpu_tensor_read_f32(y, projected, 32), "projection readback");
    for (int i = 0; i < 32; i++)
        require(fabsf(projected[i] - 15) < 1e-5f,
                "Metal adapted projection scalar oracle");
    quantization(gpu);
    parallel_loader(gpu);
    h3_gpu_tensor_free(x);
    h3_gpu_tensor_free(y);
    h3_gpu_tensor_free(w);
    h3_gpu_free(gpu);
    free(input);
    free(readback);
    h3_lora_stack_free(r);
    h3_weight_store_free(store);
    h3_lora_stack_free(s);
    store = h3_weight_store_open(dir, e, sizeof(e));
    require(store != NULL, e);
    adapter(ap, "final_layer.audio_out", 1, -1, 1, 1);
    h3_lora_spec invalid = {ap, 1};
    require(!h3_lora_stack_load(&invalid, 1, store, e, sizeof(e)),
            "negative alpha rejected");
    adapter(ap, "final_layer.audio_out", 1, 0, NAN, 1);
    require(!h3_lora_stack_load(&invalid, 1, store, e, sizeof(e)),
            "zero alpha still validates nonfinite payload");
    adapter(ap, "unknown_target", 0, 0, 1, 1);
    require(!h3_lora_stack_load(&invalid, 1, store, e, sizeof(e)),
            "unknown target rejected");
    float scalar = 1;
    file(ap,
         "{\"final_layer.audio_out.alpha\":{\"dtype\":\"F32\",\"shape\":[],\"data_"
         "offsets\":[0,4]}}",
         &scalar, 4);
    require(!h3_lora_stack_load(&invalid, 1, store, e, sizeof(e)) &&
                strstr(e, "orphan"),
            "orphan alpha rejected");
    file(ap,
         "{\"final_layer.audio_out.lora_A.weight\":{\"dtype\":\"F32\",\"shape\":[1,"
         "5376],\"data_offsets\":[0,21504]}}",
         base, 21504);
    require(!h3_lora_stack_load(&invalid, 1, store, e, sizeof(e)) &&
                strstr(e, "missing"),
            "missing pair rejected");
    file(ap,
         "{\"final_layer.audio_out.lora_A.weight\":{\"dtype\":\"F32\",\"shape\":[1,1],"
         "\"data_offsets\":[0,4]},\"final_layer.audio_out.lora_B.weight\":{\"dtype\":"
         "\"F32\",\"shape\":[32,1],\"data_offsets\":[4,132]}}",
         base, 132);
    require(!h3_lora_stack_load(&invalid, 1, store, e, sizeof(e)) && strstr(e, "shape"),
            "wrong dimensions rejected");
    file(ap,
         "{\"final_layer.audio_out.lora_A.weight\":{\"dtype\":\"F16\",\"shape\":[1,"
         "5376],\"data_offsets\":[0,10752]},\"final_layer.audio_out.lora_B.weight\":{" 
         "\"dtype\":\"F32\",\"shape\":[32,1],\"data_offsets\":[10752,10880]}}",
         base, 10880);
    h3_lora_stack *half_stack = h3_lora_stack_load(&invalid, 1, store, e, sizeof(e));
    require(half_stack != NULL, e);
    h3_lora_stack_free(half_stack);
    file(ap, "{\"__metadata__\":{\"application\":\"W_eff = W + B @ A / sqrt(rank)\"}}",
         NULL, 0);
    require(!h3_lora_stack_load(&invalid, 1, store, e, sizeof(e)) &&
                strstr(e, "application"),
            "conflicting scaling metadata rejected");
    adapter(ap, "final_layer.audio_out", 0, 0, 1, 1);
    require(!truncate(ap, 8), "truncate fixture");
    require(!h3_lora_stack_load(&invalid, 1, store, e, sizeof(e)),
            "truncated artifact rejected atomically");
    adapter(ap, "final_layer.audio_out", 0, 0, 1, 1);
    h3_weight_store *wrong = h3_weight_store_open(dir, e, sizeof(e));
    require(wrong != NULL, e);
    h3_lora_stack *valid = h3_lora_stack_load(&invalid, 1, wrong, e, sizeof(e));
    require(valid != NULL, e);
    h3_gpu_tensor *already =
        h3_weight_load_f32(wrong, NULL, "absent", 2, shape, e, sizeof(e));
    require(!already && !h3_weight_store_bind_loras(wrong, valid, e, sizeof(e)),
            "binding after payload load attempt rejected");
    h3_weight_store_free(wrong);
    h3_lora_stack_free(valid);
    invalid.strength = INFINITY;
    require(!h3_lora_stack_load(&invalid, 1, store, e, sizeof(e)),
            "nonfinite strength rejected");
    free(row);
    free(other);
    free(base);
    h3_weight_store_free(store);
    unlink(ap);
    unlink(bp);
    require(!rmdir(dir), "fixture cleanup");
}
int main(void) {
    numeric();
    parser_validation();
    stack_and_metal();
    puts("LoRA numeric, validation, snapshot, canonical and Metal projection tests "
         "passed");
    return 0;
}
