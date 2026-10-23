#ifndef H3_LORA_H
#define H3_LORA_H
#include "h3_weights.h"
#include <stdio.h>
typedef struct h3_lora_stack h3_lora_stack;
/* Typed production composition seam; factors remain in their original dtype. */
typedef struct {
    const void *a, *b;
    h3_dtype a_dtype, b_dtype;
    size_t rank;
    float scale;
} h3_lora_factor;
typedef struct h3_lora_target {
    char *name;
    size_t rows, columns;
    h3_dtype dtype;
    size_t heads, head_dim; /* both zero for identity output layout */
    h3_lora_factor *factors;
    size_t factor_count;
} h3_lora_target;
h3_lora_stack *h3_lora_stack_load(const h3_lora_spec *, size_t, const h3_weight_store *,
                                  char *, size_t);
void h3_lora_stack_free(h3_lora_stack *);
const h3_lora_spec *h3_lora_stack_specs(const h3_lora_stack *, size_t *);
const char *h3_lora_stack_identity(const h3_lora_stack *);
void h3_lora_stack_report(const h3_lora_stack *, FILE *);
const char *h3_lora_stack_base(const h3_lora_stack *);
const h3_lora_target *h3_lora_find_target(const h3_lora_stack *, const char *);
size_t h3_lora_scratch_bytes(const h3_lora_target *, size_t);
int h3_lora_merge_rows(const h3_lora_target *, uint64_t, size_t, void *, void *, size_t,
                       char *, size_t);
h3_gpu_tensor *h3_lora_load_target(const h3_lora_target *, h3_gpu *,
                                   const h3_st_header *, const h3_st_tensor *, char *,
                                   size_t);
#endif
