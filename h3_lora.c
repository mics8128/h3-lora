#define ACCELERATE_NEW_LAPACK
#include "h3_lora.h"
#include <Accelerate/Accelerate.h>
#include <CommonCrypto/CommonDigest.h>
#include <dispatch/dispatch.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define ROWS 128u
#define RANK 64u

typedef struct {
    unsigned char *bytes;
    size_t size;
    unsigned char hash[32];
    char hex[65];
    h3_st_header header;
    size_t pairs, default_alpha;
    size_t min_rank, max_rank;
    int has_metadata_alpha;
    float metadata_alpha;
    size_t alpha_conflicts;
    int dynamic_alpha, baked_alpha;
} backing;
typedef struct {
    backing *data;
    h3_lora_spec spec;
} entry;
struct h3_lora_stack {
    char *base;
    backing **backings;
    size_t backing_count;
    entry *entries;
    h3_lora_spec *specs;
    size_t count;
    h3_lora_target *targets;
    size_t target_count;
    char identity[65];
};
static int fail(char *e, size_t n, const char *fmt, ...) {
    if (e && n) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(e, n, fmt, ap);
        va_end(ap);
    }
    return 0;
}
static void hex_digest(const unsigned char *digest, char *hex) {
    static const char digits[] = "0123456789abcdef";
    for (size_t i = 0; i < 32; i++) {
        hex[2 * i] = digits[digest[i] >> 4];
        hex[2 * i + 1] = digits[digest[i] & 15];
    }
    hex[64] = 0;
}
static void hash_bytes(const void *p, size_t n, unsigned char out[32]) {
    CC_SHA256_CTX c;
    CC_SHA256_Init(&c);
    while (n) {
        CC_LONG k = n > UINT_MAX ? UINT_MAX : (CC_LONG)n;
        CC_SHA256_Update(&c, p, k);
        p = (const char *)p + k;
        n -= k;
    }
    CC_SHA256_Final(out, &c);
}
static float value(const void *data, h3_dtype dtype, size_t i) {
    float f;
    if (dtype == H3_DTYPE_F32)
        memcpy(&f, (const char *)data + 4 * i, 4);
    else {
        uint16_t b;
        memcpy(&b, (const char *)data + 2 * i, 2);
        uint32_t u;
        if (dtype == H3_DTYPE_F16) {
            uint32_t sign = (uint32_t)(b & 0x8000) << 16;
            unsigned exponent = (b >> 10) & 31, fraction = b & 1023;
            if (!exponent && fraction) {
                int shift = 0;
                while (!(fraction & 1024)) {
                    fraction <<= 1;
                    shift++;
                }
                u = sign | (uint32_t)(113 - shift) << 23 |
                    (uint32_t)(fraction & 1023) << 13;
            } else if (!exponent)
                u = sign;
            else if (exponent == 31)
                u = sign | 0x7f800000u | (uint32_t)fraction << 13;
            else
                u = sign | (uint32_t)(exponent + 112) << 23 |
                    (uint32_t)fraction << 13;
        } else
            u = (uint32_t)b << 16;
        memcpy(&f, &u, 4);
    }
    return f;
}
static uint16_t bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}
static int matrix_dtype(h3_dtype d) {
    return d == H3_DTYPE_F32 || d == H3_DTYPE_BF16;
}
static int factor_dtype(h3_dtype d) {
    return matrix_dtype(d) || d == H3_DTYPE_F16;
}

/* Enumerated semantic inventory also creates the collision-checked Kohya alias
 * table. Flattening is only applied to these known native names, never input. */
static int semantic(const char *source, char native[160], size_t *m, size_t *n,
                    h3_dtype *dtype, int *qkv) {
    const char *s = source;
    if (!strncmp(s, "model.diffusion_model.", 22))
        s += 22;
    else if (!strncmp(s, "diffusion_model.", 16))
        s += 16;
    struct projection {
        const char *name;
        size_t m, n;
    };
    static const struct projection p[] = {{"attn.qkv_proj", 21504, 5376},
                                          {"attn.out_proj", 5376, 7168},
                                          {"mlp.fc1", 28672, 5376},
                                          {"mlp.fc2", 5376, 14336},
                                          {"adaln_proj.linear", 96768, 2688}};
    size_t matches = 0;
    for (int group = 0; group < 2; group++)
        for (int block = 0; block < (group ? 2 : 50); block++)
            for (size_t j = 0; j < (group ? 4 : 5); j++) {
                char stem[150], alias[160];
                snprintf(stem, sizeof(stem), "%sblocks.%d.%s",
                         group ? "token_refiner." : "", block, p[j].name);
                snprintf(alias, sizeof(alias), "lora_unet_%s", stem);
                for (char *c = alias + 10; *c; c++)
                    if (*c == '.')
                        *c = '_';
                if (strcmp(s, stem) && strcmp(source, alias))
                    continue;
                snprintf(native, 160, "%s.weight", stem);
                *m = p[j].m;
                *n = p[j].n;
                *dtype = H3_DTYPE_BF16;
                *qkv = j == 0;
                matches++;
            }
    static const struct {
        const char *name;
        size_t m, n;
        h3_dtype d;
    } other[] = {{"condition_proj", 5376, 5120, H3_DTYPE_BF16},
                 {"time_embedder.proj_in", 5376, 256, H3_DTYPE_F32},
                 {"time_embedder.proj_out", 2688, 5376, H3_DTYPE_F32},
                 {"video_patch_proj", 5376, 96, H3_DTYPE_F32},
                 {"audio_patch_proj", 5376, 32, H3_DTYPE_F32},
                 {"final_layer.video_out", 96, 5376, H3_DTYPE_F32},
                 {"final_layer.audio_out", 32, 5376, H3_DTYPE_F32},
                 {"final_layer.adaln_proj.linear", 10752, 2688, H3_DTYPE_BF16}};
    for (size_t j = 0; j < sizeof(other) / sizeof(*other); j++) {
        char alias[160];
        snprintf(alias, sizeof(alias), "lora_unet_%s", other[j].name);
        for (char *c = alias + 10; *c; c++)
            if (*c == '.')
                *c = '_';
        if (strcmp(s, other[j].name) && strcmp(source, alias))
            continue;
        snprintf(native, 160, "%s.weight", other[j].name);
        *m = other[j].m;
        *n = other[j].n;
        *dtype = other[j].d;
        *qkv = 0;
        matches++;
    }
    return matches == 1;
}
static int split_key(const char *name, char stem[160], int *kind, int *family) {
    static const char *suffix[] = {".lora_down.weight", ".lora_up.weight",
                                   ".lora_A.weight", ".lora_B.weight", ".alpha"};
    size_t length = strlen(name);
    for (int index = 0; index < 5; index++) {
        size_t tail = strlen(suffix[index]);
        if (length > tail && !strcmp(name + length - tail, suffix[index])) {
            size_t size = length - tail;
            if (size >= 160)
                return 0;
            memcpy(stem, name, size);
            stem[size] = '\0';
            *kind = index == 4 ? 2 : index % 2;
            *family = index < 2 ? 0 : 1;
            return 1;
        }
    }
    return 0;
}
static int compare_entry(const void *a, const void *b) {
    const entry *x = a, *y = b;
    int c = memcmp(x->data->hash, y->data->hash, 32);
    if (c)
        return c;
    if (x->spec.strength < y->spec.strength)
        return -1;
    if (x->spec.strength > y->spec.strength)
        return 1;
    return strcmp(x->spec.path, y->spec.path);
}
static int compare_target(const void *a, const void *b) {
    return strcmp(((const h3_lora_target *)a)->name, ((const h3_lora_target *)b)->name);
}
static void backing_free(backing *b) {
    if (b) {
        h3_st_free_header(&b->header);
        free(b->bytes);
        free(b);
    }
}
static backing *snapshot(const char *path, char *e, size_t en) {
    int fd = open(path, O_RDONLY);
    struct stat st;
    if (fd < 0) {
        fail(e, en, "adapter %s: %s", path, strerror(errno));
        return NULL;
    }
    if (fstat(fd, &st) || st.st_size < 8 || (uint64_t)st.st_size > SIZE_MAX) {
        close(fd);
        fail(e, en, "adapter %s: invalid size", path);
        return NULL;
    }
    backing *b = calloc(1, sizeof(*b));
    if (!b) {
        close(fd);
        fail(e, en, "adapter snapshot allocation failed");
        return NULL;
    }
    b->size = (size_t)st.st_size;
    b->bytes = malloc(b->size);
    if (!b->bytes) {
        close(fd);
        backing_free(b);
        fail(e, en, "adapter snapshot allocation failed");
        return NULL;
    }
    size_t done = 0;
    while (done < b->size) {
        size_t n = b->size - done;
        if (n > (1u << 30))
            n = 1u << 30;
        ssize_t k = pread(fd, b->bytes + done, n, (off_t)done);
        if (k < 0 && errno == EINTR)
            continue;
        if (k <= 0) {
            close(fd);
            backing_free(b);
            fail(e, en, "adapter %s: incomplete snapshot", path);
            return NULL;
        }
        done += (size_t)k;
    }
    close(fd);
    hash_bytes(b->bytes, b->size, b->hash);
    hex_digest(b->hash, b->hex);
    char detail[512];
    if (!h3_st_read_memory(b->bytes, b->size, path, &b->header, detail,
                           sizeof(detail))) {
        fail(e, en, "adapter %s: %s", path, detail);
        backing_free(b);
        return NULL;
    }
    return b;
}
static int metadata_valid(backing *b, char *e, size_t en) {
    const char *module = NULL, *alpha = NULL, *conversion = NULL;
    const char *baked = NULL, *normalized = NULL, *normalization = NULL;
    for (size_t i = 0; i < b->header.metadata_count; i++) {
        const char *k = b->header.metadata_keys[i], *v = b->header.metadata_values[i];
        if (!strcmp(k, "ss_network_module")) module = v;
        if (!strcmp(k, "ss_network_alpha")) alpha = v;
        if (!strcmp(k, "conversion")) conversion = v;
        if (!strcmp(k, "baked_scale")) baked = v;
        if (!strcmp(k, "alpha_normalized")) normalized = v;
        if (!strcmp(k, "alpha_normalization")) normalization = v;
    }
    if (baked) {
        int known = conversion &&
            ((!strcmp(baked, "0.125") &&
              (!strcmp(conversion, "qkv fused contiguous q|k|v (block-diagonal, rank 3x); mlp.fc1 swiglu halves swapped; peft alpha/r=0.125 baked into lora_B") ||
               !strcmp(conversion, "qkv block-diag fused (per-projection ranks); mlp.fc1 swiglu halves swapped; alpha dropped (alpha==rank, scale 1.0); effective scale 0.125 baked into lora_B (flat multiplier, NOT alpha/rank)"))) ||
             (!strcmp(baked, "1.0") &&
              !strcmp(conversion, "qkv block-diag fused (contiguous q|k|v); mlp.fc1 swiglu halves swapped; alpha folded into lora_B; scale 1.0 baked as a flat multiplier")));
        if (!known)
            return fail(e, en, "adapter %s: unsupported baked scaling declaration", b->header.path);
        b->baked_alpha = 1;
    }
    if (normalized || normalization) {
        if (!normalized || strcmp(normalized, "true") || !normalization ||
            strcmp(normalization, "lora_up := lora_up * (alpha / rank); alpha tensors removed"))
            return fail(e, en, "adapter %s: unsupported alpha normalization", b->header.path);
        b->baked_alpha = 1;
    }
    b->dynamic_alpha = alpha && !strcmp(alpha, "Dynamic");
    if (alpha && !b->dynamic_alpha) {
        if (b->baked_alpha)
            return fail(e, en, "adapter %s: scalar metadata alpha conflicts with baked scaling", b->header.path);
        char *end;
        float parsed = strtof(alpha, &end);
        if (!module || strcmp(module, "networks.lora") || !*alpha ||
            end == alpha || *end || !isfinite(parsed) || parsed < 0)
            return fail(e, en, "adapter %s: unsupported/malformed ss_network_alpha=%s",
                        b->header.path, alpha);
        b->has_metadata_alpha = 1;
        b->metadata_alpha = parsed;
    }
    for (size_t i = 0; i < b->header.metadata_count; i++) {
        const char *k = b->header.metadata_keys[i], *v = b->header.metadata_values[i];
        if (!strcmp(k, "application") && strcmp(v, "W_eff = W + lora_B @ lora_A"))
            return fail(e, en, "adapter %s: unsupported application declaration %s",
                        b->header.path, v);
        if (!strcmp(k, "ss_network_module") && strcmp(v, "networks.lora"))
            return fail(e, en, "adapter %s: unsupported network module %s",
                        b->header.path, v);
        if (!strcmp(k, "converted_layout") && !strcmp(v, "comfyui_minimax_h3"))
            continue;
        if (strstr(k, "dora") || strstr(k, "rslora") || strstr(k, "scaling") ||
            strstr(k, "layout") || !strcmp(k, "ss_network_args"))
            return fail(e, en, "adapter %s: unsupported scaling/layout metadata %s=%s",
                        b->header.path, k, v);
    }
    return 1;
}
static int append_adapter(h3_lora_stack *stack, entry *ent, const h3_weight_store *base,
                          char *error, size_t error_size) {
    backing *data = ent->data;
    const h3_st_header *header = &data->header;
    unsigned char *seen = calloc(header->tensor_count, 1);
    const char **resolved = calloc(header->tensor_count, sizeof(*resolved));
    if (!seen || !resolved) {
        free(seen);
        free(resolved);
        return fail(error, error_size, "adapter indexing allocation failed");
    }
    size_t pairs = 0, defaults = 0, conflicts = 0;
    for (size_t index = 0; index < header->tensor_count; index++) {
        if (seen[index])
            continue;
        char stem[160];
        int kind, family;
        const h3_st_tensor *first = &header->tensors[index];
        if (!split_key(first->name, stem, &kind, &family)) {
            fail(error, error_size, "adapter %s key %s: unsupported update kind",
                 ent->spec.path, first->name);
            goto invalid;
        }
        char native[160];
        size_t rows, columns;
        h3_dtype dtype;
        int qkv;
        if (!semantic(stem, native, &rows, &columns, &dtype, &qkv)) {
            fail(error, error_size,
                 "adapter %s key %s: unsupported/ambiguous target %s", ent->spec.path,
                 first->name, stem);
            goto invalid;
        }
        for (size_t previous = 0; previous < pairs; previous++) {
            if (!strcmp(resolved[previous], native)) {
                fail(error, error_size, "adapter %s target %s: duplicate aliases",
                     ent->spec.path, native);
                goto invalid;
            }
        }
        const h3_st_tensor *a = NULL, *b = NULL, *alpha = NULL;
        int a_family = -1, b_family = -1;
        for (size_t other_index = 0; other_index < header->tensor_count;
             other_index++) {
            char other[160];
            int other_kind, other_family;
            if (!split_key(header->tensors[other_index].name, other, &other_kind,
                           &other_family) ||
                strcmp(stem, other))
                continue;
            if ((other_kind == 0 && a) || (other_kind == 1 && b) ||
                (other_kind == 2 && alpha)) {
                fail(error, error_size,
                     "adapter %s target %s: duplicate update aliases", ent->spec.path,
                     native);
                goto invalid;
            }
            if (other_kind == 0) {
                a = &header->tensors[other_index];
                a_family = other_family;
            } else if (other_kind == 1) {
                b = &header->tensors[other_index];
                b_family = other_family;
            } else
                alpha = &header->tensors[other_index];
            seen[other_index] = 1;
        }
        if (!a || !b) {
            fail(error, error_size,
                 "adapter %s key %s target %s: missing A/B partner or orphan alpha",
                 ent->spec.path, first->name, native);
            goto invalid;
        }
        if (a_family != b_family) {
            fail(error, error_size, "adapter %s target %s: mixed A/B families",
                 ent->spec.path, native);
            goto invalid;
        }
        const h3_st_tensor *base_tensor = h3_weight_find(base, native, NULL);
        if (!base_tensor || base_tensor->ndim != 2 || base_tensor->dtype != dtype ||
            base_tensor->shape[0] != rows || base_tensor->shape[1] != columns) {
            fail(error, error_size,
                 "adapter %s key %s target %s: incompatible or absent base schema",
                 ent->spec.path, first->name, native);
            goto invalid;
        }
        if (a->ndim != 2 || b->ndim != 2 || !factor_dtype(a->dtype) ||
            !factor_dtype(b->dtype) || !a->shape[0] || a->shape[0] > INT_MAX ||
            a->shape[1] != columns || b->shape[0] != rows ||
            b->shape[1] != a->shape[0]) {
            fail(error, error_size,
                 "adapter %s key %s target %s: invalid pair dtype/rank/shape",
                 ent->spec.path, first->name, native);
            goto invalid;
        }
        size_t rank = (size_t)a->shape[0];
        if (data->dynamic_alpha && !alpha && !data->baked_alpha) {
            fail(error, error_size, "adapter %s target %s: Dynamic alpha requires tensor alpha or declared baked normalization",
                 ent->spec.path, native);
            goto invalid;
        }
        float alpha_value = data->has_metadata_alpha ? data->metadata_alpha : (float)rank;
        if (!data->min_rank || rank < data->min_rank)
            data->min_rank = rank;
        if (rank > data->max_rank)
            data->max_rank = rank;
        if (alpha) {
            if (!factor_dtype(alpha->dtype) || h3_st_tensor_elements(alpha) != 1 ||
                alpha->ndim > 1) {
                fail(error, error_size,
                     "adapter %s target %s: alpha must be scalar F32/BF16/F16",
                     ent->spec.path, native);
                goto invalid;
            }
            alpha_value = value(data->bytes + alpha->file_offset, alpha->dtype, 0);
            if (data->has_metadata_alpha && alpha_value != data->metadata_alpha)
                conflicts++;
        } else if (!data->has_metadata_alpha)
            defaults++;
        float scale = ent->spec.strength * (alpha_value / (float)rank);
        if (!isfinite(alpha_value) || alpha_value < 0 || !isfinite(scale)) {
            fail(error, error_size,
                 "adapter %s target %s: nonfinite/negative alpha or scale",
                 ent->spec.path, native);
            goto invalid;
        }
        const void *a_values = data->bytes + a->file_offset;
        const void *b_values = data->bytes + b->file_offset;
        for (size_t element = 0; element < rank * columns; element++) {
            if (!isfinite(value(a_values, a->dtype, element))) {
                fail(error, error_size, "adapter %s key %s target %s: nonfinite A",
                     ent->spec.path, a->name, native);
                goto invalid;
            }
        }
        for (size_t element = 0; element < rows * rank; element++) {
            if (!isfinite(value(b_values, b->dtype, element))) {
                fail(error, error_size, "adapter %s key %s target %s: nonfinite B",
                     ent->spec.path, b->name, native);
                goto invalid;
            }
        }
        h3_lora_target *target = NULL;
        for (size_t candidate = 0; candidate < stack->target_count; candidate++) {
            if (!strcmp(native, stack->targets[candidate].name)) {
                target = &stack->targets[candidate];
                break;
            }
        }
        if (!target) {
            h3_lora_target *grown =
                realloc(stack->targets, (stack->target_count + 1) * sizeof(*grown));
            if (!grown) {
                fail(error, error_size, "target allocation failed");
                goto invalid;
            }
            stack->targets = grown;
            target = &grown[stack->target_count++];
            memset(target, 0, sizeof(*target));
            target->name = strdup(native);
            if (!target->name) {
                fail(error, error_size, "target allocation failed");
                goto invalid;
            }
            target->rows = rows;
            target->columns = columns;
            target->dtype = dtype;
            target->heads = qkv ? 56 : 0;
            target->head_dim = qkv ? 128 : 0;
        }
        h3_lora_factor *factors =
            realloc(target->factors, (target->factor_count + 1) * sizeof(*factors));
        if (!factors) {
            fail(error, error_size, "factor allocation failed");
            goto invalid;
        }
        target->factors = factors;
        factors[target->factor_count++] =
            (h3_lora_factor){a_values, b_values, a->dtype, b->dtype, rank, scale};
        resolved[pairs++] = target->name;
    }
    free(resolved);
    free(seen);
    if (!pairs)
        return fail(error, error_size, "adapter %s: no supported pairs",
                    ent->spec.path);
    data->pairs = pairs;
    data->default_alpha = defaults;
    data->alpha_conflicts = conflicts;
    return 1;
invalid:
    free(resolved);
    free(seen);
    return 0;
}
h3_lora_stack *h3_lora_stack_load(const h3_lora_spec *specs, size_t count,
                                  const h3_weight_store *base, char *e, size_t en) {
    if (e && en)
        e[0] = 0;
    if ((count && !specs) || !base) {
        fail(e, en, "invalid LoRA stack request");
        return NULL;
    }
    size_t active = 0;
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(specs[i].strength)) {
            fail(e, en, "nonfinite LoRA strength");
            return NULL;
        }
        if (specs[i].strength != 0)
            active++;
    }
    if (!active)
        return NULL;
    if (active > SIZE_MAX / sizeof(entry) || active > SIZE_MAX / sizeof(h3_lora_spec) ||
        active > SIZE_MAX / sizeof(backing *)) {
        fail(e, en, "LoRA count overflow");
        return NULL;
    }
    h3_lora_stack *s = calloc(1, sizeof(*s));
    if (!s) {
        fail(e, en, "stack allocation failed");
        return NULL;
    }
    const char *directory = h3_weight_store_directory(base);
    if (!directory) {
        fail(e, en, "cannot resolve base component identity");
        goto invalid;
    }
    s->base = strdup(directory);
    s->entries = calloc(active, sizeof(entry));
    s->specs = calloc(active, sizeof(h3_lora_spec));
    s->backings = calloc(active, sizeof(backing *));
    if (!s->base || !s->entries || !s->specs || !s->backings) {
        fail(e, en, "stack allocation failed");
        goto invalid;
    }
    for (size_t i = 0; i < count; i++)
        if (specs[i].strength != 0) {
            if (!specs[i].path || !*specs[i].path) {
                fail(e, en, "active adapter path required");
                goto invalid;
            }
            backing *b = NULL;
            for (size_t previous = 0; previous < s->count; previous++) {
                if (!strcmp(specs[i].path, s->entries[previous].spec.path)) {
                    b = s->entries[previous].data;
                    break;
                }
            }
            if (!b) {
                b = snapshot(specs[i].path, e, en);
                if (!b)
                    goto invalid;
                backing *shared = NULL;
                for (size_t previous = 0; previous < s->backing_count; previous++) {
                    backing *candidate = s->backings[previous];
                    if (b->size == candidate->size &&
                        !memcmp(b->hash, candidate->hash, 32) &&
                        !memcmp(b->bytes, candidate->bytes, b->size)) {
                        shared = candidate;
                        break;
                    }
                }
                if (shared) {
                    backing_free(b);
                    b = shared;
                } else {
                    s->backings[s->backing_count++] = b;
                    if (!metadata_valid(b, e, en))
                        goto invalid;
                }
            }
            entry *ent = &s->entries[s->count++];
            ent->data = b;
            ent->spec.strength = specs[i].strength;
            ent->spec.path = strdup(specs[i].path);
            if (!ent->spec.path) {
                fail(e, en, "path allocation failed");
                goto invalid;
            }
        }
    qsort(s->entries, s->count, sizeof(entry), compare_entry);
    CC_SHA256_CTX hash;
    CC_SHA256_Init(&hash);
    static const char dialect[] =
        "H3-Comfy-linear-v1:qkv-contiguous-to-head56x128:fc1-gate-up:alpha-rank";
    CC_SHA256_Update(&hash, dialect, (CC_LONG)(sizeof(dialect) - 1));
    for (size_t i = 0; i < s->count; i++) {
        entry *ent = &s->entries[i];
        s->specs[i] = ent->spec;
        if (!append_adapter(s, ent, base, e, en))
            goto invalid;
        CC_SHA256_Update(&hash, ent->data->hash, 32);
        uint32_t u;
        memcpy(&u, &ent->spec.strength, 4);
        unsigned char le[4] = {(unsigned char)u, (unsigned char)(u >> 8),
                               (unsigned char)(u >> 16), (unsigned char)(u >> 24)};
        CC_SHA256_Update(&hash, le, 4);
    }
    unsigned char digest[32];
    CC_SHA256_Final(digest, &hash);
    hex_digest(digest, s->identity);
    qsort(s->targets, s->target_count, sizeof(*s->targets), compare_target);
    return s;
invalid:
    h3_lora_stack_free(s);
    return NULL;
}
void h3_lora_stack_free(h3_lora_stack *s) {
    if (!s)
        return;
    for (size_t i = 0; i < s->count; i++)
        free((char *)s->entries[i].spec.path);
    for (size_t i = 0; i < s->backing_count; i++)
        backing_free(s->backings[i]);
    for (size_t i = 0; i < s->target_count; i++) {
        free(s->targets[i].name);
        free(s->targets[i].factors);
    }
    free(s->base);
    free(s->entries);
    free(s->specs);
    free(s->backings);
    free(s->targets);
    free(s);
}
const h3_lora_spec *h3_lora_stack_specs(const h3_lora_stack *s, size_t *n) {
    if (n)
        *n = s ? s->count : 0;
    return s ? s->specs : NULL;
}
const char *h3_lora_stack_identity(const h3_lora_stack *s) {
    return s ? s->identity : "";
}
const char *h3_lora_stack_base(const h3_lora_stack *s) {
    return s ? s->base : NULL;
}
const h3_lora_target *h3_lora_find_target(const h3_lora_stack *s, const char *name) {
    if (s && name)
        for (size_t i = 0; i < s->target_count; i++)
            if (!strcmp(s->targets[i].name, name))
                return &s->targets[i];
    return NULL;
}
void h3_lora_stack_report(const h3_lora_stack *s, FILE *out) {
    if (!s || !out)
        return;
    size_t bytes = 0;
    for (size_t i = 0; i < s->backing_count; i++)
        bytes += s->backings[i]->size;
    fprintf(out, "LoRA: %zu active adapters; %zu resident snapshot bytes\n", s->count,
            bytes);
    for (size_t i = 0; i < s->count; i++) {
        const entry *ent = &s->entries[i];
        const backing *b = ent->data;
        fprintf(
            out,
            "  %s strength %.9g SHA256 %s; H3 Comfy linear; "
            "%zu matched pairs/targets; rank %zu..%zu; "
            "scale=strength*alpha/rank, absent alpha defaults to rank (%zu targets)\n",
            ent->spec.path, ent->spec.strength, b->hex, b->pairs, b->min_rank,
            b->max_rank, b->default_alpha);
        if (b->has_metadata_alpha)
            fprintf(out, "    ss_network_alpha=%.9g fallback; %zu explicit tensor "
                         "conflicts overridden by tensor alpha.\n",
                    b->metadata_alpha, b->alpha_conflicts);
        if (b->baked_alpha)
            fprintf(out, "    Declared baked scaling already in factors; not applied again.\n");
        for (size_t m = 0; m < b->header.metadata_count; m++) {
            if (!strcmp(b->header.metadata_keys[m], "partial_conversion") &&
                !strcmp(b->header.metadata_values[m], "true"))
                fprintf(out, "    Partial conversion: full-sized target compatibility "
                             "does not establish four-step distillation quality.\n");
        }
    }
    fprintf(out,
            "  Immutable for this context; bound to validated transformer; SSD streaming unsupported.\n");
}
size_t h3_lora_scratch_bytes(const h3_lora_target *t, size_t rows) {
    if (!t || rows > ROWS || !rows ||
        t->columns > SIZE_MAX / (sizeof(float) * (rows + RANK)))
        return 0;
    size_t floats = (rows + RANK) * t->columns + rows * RANK;
    if (floats > SIZE_MAX / sizeof(float))
        return 0;
    return floats * sizeof(float);
}
int h3_lora_merge_rows(const h3_lora_target *t, uint64_t first, size_t rows,
                       void *native, void *scratch, size_t bytes, char *e, size_t en) {
    size_t need = h3_lora_scratch_bytes(t, rows);
    if (!need || !native || !scratch || bytes < need || first > t->rows ||
        rows > t->rows - first || t->columns > INT_MAX || !matrix_dtype(t->dtype) ||
        (t->heads && (!t->head_dim || t->heads > SIZE_MAX / t->head_dim / 3 ||
                      t->rows != 3 * t->heads * t->head_dim)))
        return fail(e, en, "invalid composition tile");
    size_t n = t->columns;
    float *delta = scratch, *a = delta + rows * n, *b = a + RANK * n;
    memset(delta, 0, rows * n * sizeof(float));
    for (size_t f = 0; f < t->factor_count; f++) {
        const h3_lora_factor *p = &t->factors[f];
        if (!p->rank || !factor_dtype(p->a_dtype) || !factor_dtype(p->b_dtype) ||
            !isfinite(p->scale))
            return fail(e, en, "invalid composition factor");
        for (size_t r = 0; r < p->rank; r += RANK) {
            size_t k = p->rank - r;
            if (k > RANK)
                k = RANK;
            const float *ap;
            if (p->a_dtype == H3_DTYPE_F32 && (uintptr_t)p->a % _Alignof(float) == 0) {
                ap = (const float *)p->a + r * n;
            } else {
                for (size_t j = 0; j < k * n; j++)
                    a[j] = value(p->a, p->a_dtype, r * n + j);
                ap = a;
            }
            for (size_t i = 0; i < rows; i++) {
                size_t src = (size_t)first + i;
                if (t->heads) {
                    size_t head = src / (3 * t->head_dim), q = (src / t->head_dim) % 3,
                           d = src % t->head_dim;
                    src = (q * t->heads + head) * t->head_dim + d;
                }
                for (size_t j = 0; j < k; j++)
                    b[i * k + j] = value(p->b, p->b_dtype, src * p->rank + r + j);
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)rows, (int)n,
                        (int)k, p->scale, b, (int)k, ap, (int)n, 1, delta, (int)n);
        }
    }
    for (size_t i = 0; i < rows * n; i++) {
        float result = value(native, t->dtype, i) + delta[i];
        if (!isfinite(result))
            return fail(e, en, "nonfinite merged result in %s",
                        t->name ? t->name : "target");
        if (t->dtype == H3_DTYPE_F32)
            memcpy((char *)native + 4 * i, &result, 4);
        else {
            uint16_t v = bf16(result);
            if ((v & 0x7f80) == 0x7f80)
                return fail(e, en, "BF16 overflow in %s", t->name ? t->name : "target");
            memcpy((char *)native + 2 * i, &v, 2);
        }
    }
    return 1;
}
typedef struct {
    const h3_lora_target *target;
    const h3_st_tensor *base;
    h3_gpu_tensor *result;
    int fd, success;
    size_t first_row, end_row, scratch_bytes, item;
    void *scratch, *tile;
    char error[512];
} lora_load_job;

static void load_target_rows(void *context, size_t index) {
    lora_load_job *job = &((lora_load_job *)context)[index];
    const h3_lora_target *t = job->target;
    for (size_t row = job->first_row; row < job->end_row; row += ROWS) {
        size_t count = t->rows - row;
        if (count > ROWS) count = ROWS;
        size_t bytes = count * t->columns * job->item, done = 0;
        while (done < bytes) {
            ssize_t n = pread(job->fd, (char *)job->tile + done, bytes - done,
                (off_t)(job->base->file_offset + row * t->columns * job->item + done));
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) {
                fail(job->error, sizeof(job->error), "cannot read base tile %s", job->base->name);
                return;
            }
            done += (size_t)n;
        }
        if (!h3_lora_merge_rows(t, row, count, job->tile, job->scratch,
                job->scratch_bytes, job->error, sizeof(job->error))) return;
        int ok = t->dtype == H3_DTYPE_BF16 ?
            h3_gpu_tensor_write_bf16_range(job->result, row * t->columns,
                                           job->tile, count * t->columns) :
            h3_gpu_tensor_write_f32_range(job->result, row * t->columns,
                                          job->tile, count * t->columns);
        if (!ok) {
            fail(job->error, sizeof(job->error), "cannot upload adapted tile %s", job->base->name);
            return;
        }
    }
    job->success = 1;
}

h3_gpu_tensor *h3_lora_load_target(const h3_lora_target *t, h3_gpu *gpu,
                                   const h3_st_header *h, const h3_st_tensor *base,
                                   char *e, size_t en) {
    if (!t || !h || !base || !t->columns || !matrix_dtype(t->dtype) ||
        base->ndim != 2 || base->dtype != t->dtype ||
        base->shape[0] != t->rows || base->shape[1] != t->columns ||
        t->rows > SIZE_MAX / t->columns) {
        fail(e, en, "composition base schema changed");
        return NULL;
    }
    size_t elements = t->rows * t->columns, item = h3_dtype_size(t->dtype);
    size_t scratch_n = h3_lora_scratch_bytes(t, ROWS);
    if (!scratch_n || t->columns > SIZE_MAX / ROWS / item) {
        fail(e, en, "composition workspace size overflow");
        return NULL;
    }
    size_t tile_bytes = ROWS * t->columns * item;
    if (scratch_n > SIZE_MAX - tile_bytes) {
        fail(e, en, "composition workspace size overflow");
        return NULL;
    }
    size_t tiles = t->rows / ROWS + (t->rows % ROWS != 0);
    /* Two independent Accelerate calls avoid unbounded nested BLAS work.
     * Small targets remain serial; additional workspace is capped at 64 MiB. */
    size_t workers = tiles >= 4 && elements >= (16u * 1024u * 1024u) / item &&
        scratch_n + tile_bytes <= (64u * 1024u * 1024u) / 2 ? 2 : 1;
    lora_load_job jobs[2] = {{0}};
    h3_gpu_tensor *result = NULL;
    int fd = -1;
    for (size_t i = 0; i < workers; i++) {
        jobs[i].scratch = malloc(scratch_n);
        jobs[i].tile = malloc(tile_bytes);
        if (!jobs[i].scratch || !jobs[i].tile) {
            fail(e, en, "composition workspace allocation failed");
            goto invalid;
        }
    }
    result = t->dtype == H3_DTYPE_BF16 ? h3_gpu_tensor_new_bf16(gpu, elements)
                                       : h3_gpu_tensor_new_f32(gpu, elements);
    if (!result) {
        fail(e, en, "adapted GPU allocation failed: %s", h3_gpu_error(gpu));
        goto invalid;
    }
    fd = open(h->path, O_RDONLY);
    if (fd < 0) {
        fail(e, en, "cannot open base tensor %s", base->name);
        goto invalid;
    }
    for (size_t i = 0; i < workers; i++) {
        jobs[i].target = t;
        jobs[i].base = base;
        jobs[i].result = result;
        jobs[i].fd = fd;
        jobs[i].item = item;
        jobs[i].scratch_bytes = scratch_n;
        jobs[i].first_row = (tiles * i / workers) * ROWS;
        jobs[i].end_row = (tiles * (i + 1) / workers) * ROWS;
    }
    /* Only disjoint shared-buffer writes run concurrently. No GPU allocation,
     * command submission or callback occurs until all row workers have joined. */
    if (workers == 1) load_target_rows(jobs, 0);
    else dispatch_apply_f(workers,
        dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), jobs, load_target_rows);
    for (size_t i = 0; i < workers; i++) {
        if (!jobs[i].success) {
            fail(e, en, "%s", jobs[i].error);
            goto invalid;
        }
    }
    close(fd);
    for (size_t i = 0; i < workers; i++) {
        free(jobs[i].scratch);
        free(jobs[i].tile);
    }
    return result;
invalid:
    if (fd >= 0)
        close(fd);
    h3_gpu_tensor_free(result);
    for (size_t i = 0; i < workers; i++) {
        free(jobs[i].scratch);
        free(jobs[i].tile);
    }
    return NULL;
}
