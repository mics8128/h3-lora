#include "h3_internal.h"
#include "h3_audio_vae.h"
#include "h3_host.h"
#include "h3_dit.h"
#include "h3_ffmpeg.h"
#include "h3_metal.h"
#include "h3_multimodal.h"
#include "h3_safetensors.h"
#include "h3_text_encoder.h"
#include "h3_tokenizer.h"
#include "h3_video_encoder.h"
#include "h3_video_vae.h"
#include "h3_vision_encoder.h"
#include "h3_lora.h"
#include "h3_weights.h"
#include "h3_sampler.h"
#include "h3_latent.h"
#include <CommonCrypto/CommonDigest.h>

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <ctype.h>
#include <fcntl.h>
#include <mach-o/dyld.h>

static char h3_global_error[512];
static char h3_global_error_code[64];
static char h3_global_error_field[128];
static h3_result *h3_decode_latent_output(h3_ctx *,const char *,int,const h3_params *,int,int);
typedef struct h3_file_digest {
    struct h3_file_digest *next;
    char *path;
    struct stat signature;
    char hex[65];
} h3_file_digest;

static int h3_same_file(const struct stat *a, const struct stat *b) {
    return a->st_dev == b->st_dev && a->st_ino == b->st_ino &&
        a->st_size == b->st_size &&
        a->st_mtimespec.tv_sec == b->st_mtimespec.tv_sec &&
        a->st_mtimespec.tv_nsec == b->st_mtimespec.tv_nsec &&
        a->st_ctimespec.tv_sec == b->st_ctimespec.tv_sec &&
        a->st_ctimespec.tv_nsec == b->st_ctimespec.tv_nsec;
}

static const char *h3_file_hash(h3_ctx *ctx, const char *path) {
    struct stat before, after, named;
    if (stat(path, &before) || !S_ISREG(before.st_mode)) return NULL;
    h3_file_digest *entry = ctx->file_digests;
    for (; entry; entry = entry->next)
        if (!strcmp(entry->path, path)) break;
    if (entry && h3_same_file(&before, &entry->signature)) return entry->hex;
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return NULL;
    int ok = !fstat(fd, &after) && h3_same_file(&before, &after);
    CC_SHA256_CTX hash;
    CC_SHA256_Init(&hash);
    unsigned char buffer[65536], digest[32];
    off_t total = 0;
    while (ok) {
        ssize_t n = read(fd, buffer, sizeof(buffer));
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) { ok = 0; break; }
        if (!n) break;
        total += n;
        CC_SHA256_Update(&hash, buffer, (CC_LONG)n);
    }
    ok = ok && total == before.st_size && !fstat(fd, &after) &&
        h3_same_file(&before, &after) && !stat(path, &named) &&
        h3_same_file(&before, &named);
    close(fd);
    if (!ok) return NULL;
    CC_SHA256_Final(digest, &hash);
    if (!entry) {
        entry = calloc(1, sizeof(*entry));
        if (!entry) return NULL;
        entry->path = strdup(path);
        if (!entry->path) { free(entry); return NULL; }
        entry->next = ctx->file_digests;
        ctx->file_digests = entry;
    }
    entry->signature = before;
    ctx->file_digest_revision++;
    static const char digits[] = "0123456789abcdef";
    for (size_t i = 0; i < sizeof(digest); i++) {
        entry->hex[2*i] = digits[digest[i] >> 4];
        entry->hex[2*i+1] = digits[digest[i] & 15];
    }
    entry->hex[64] = 0;
    return entry->hex;
}

typedef struct {
    char *text;
    size_t length;
    size_t capacity;
    h3_ctx *content_ctx;
} h3_key;

static void h3_conditioning_cache_clear(h3_ctx *ctx) {
    if (!ctx) return;
    free(ctx->conditioning_key);
    free(ctx->conditioning_values);
    free(ctx->conditioning_tags);
    free(ctx->conditioning_video_rows);
    free(ctx->conditioning_audio_rows);
    free(ctx->conditioning_references);
    free(ctx->conditioning_guides);
    ctx->conditioning_guides = NULL;
    ctx->conditioning_guide_count = 0;
    ctx->conditioning_key = NULL;
    ctx->conditioning_values = NULL;
    ctx->conditioning_tags = NULL;
    ctx->conditioning_video_rows = NULL;
    ctx->conditioning_audio_rows = NULL;
    ctx->conditioning_references = NULL;
    ctx->conditioning_tokens = 0;
    ctx->conditioning_width = 0;
    ctx->conditioning_video_elements = 0;
    ctx->conditioning_audio_elements = 0;
    ctx->conditioning_reference_count = 0;
    ctx->conditioning_present = 0;
}

static void h3_prepared_cache_clear(h3_ctx *ctx) {
    h3_dit_free(ctx->dit);
    ctx->dit = NULL;
    free(ctx->dit_key); ctx->dit_key = NULL;
    free(ctx->dit_model_key); ctx->dit_model_key = NULL;
}

void h3_cache_clear(h3_ctx *ctx) {
    if (!ctx) return;
    h3_conditioning_cache_clear(ctx);
    h3_prepared_cache_clear(ctx);
    h3_video_vae_decoder_free(ctx->video_decoder);
    ctx->video_decoder = NULL;
    free(ctx->video_decoder_key);
    ctx->video_decoder_key = NULL;
}

void h3_cache_set_enabled(h3_ctx *ctx, int enabled) {
    if (!ctx) return;
    if (!enabled) h3_cache_clear(ctx);
    ctx->cache_enabled = enabled != 0;
}

void h3_cache_get_info(const h3_ctx *ctx, h3_cache_info *info) {
    if (!info) return;
    memset(info, 0, sizeof(*info));
    if (!ctx) return;
    if (ctx->conditioning_key) {
        info->embedding_entries = 1;
        info->embedding_bytes =
            ctx->conditioning_tokens * ctx->conditioning_width *
                sizeof(*ctx->conditioning_values) +
            ctx->conditioning_tokens * sizeof(*ctx->conditioning_tags) +
            ctx->conditioning_video_elements *
                sizeof(*ctx->conditioning_video_rows) +
            ctx->conditioning_audio_elements *
                sizeof(*ctx->conditioning_audio_rows);
    }
    info->prepared_dit = ctx->dit != NULL;
    info->video_decoder = ctx->video_decoder != NULL;
}

static int h3_key_append(h3_key *key, const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    va_list copy;
    va_copy(copy, arguments);
    int needed = vsnprintf(NULL, 0, format, copy);
    va_end(copy);
    if (needed < 0) {
        va_end(arguments);
        return 0;
    }
    size_t wanted = key->length + (size_t)needed + 1;
    if (wanted > key->capacity) {
        size_t capacity = key->capacity ? key->capacity : 256;
        while (capacity < wanted) {
            if (capacity > SIZE_MAX / 2) {
                va_end(arguments);
                return 0;
            }
            capacity *= 2;
        }
        char *grown = realloc(key->text, capacity);
        if (!grown) {
            va_end(arguments);
            return 0;
        }
        key->text = grown;
        key->capacity = capacity;
    }
    vsnprintf(key->text + key->length, key->capacity - key->length,
              format, arguments);
    va_end(arguments);
    key->length += (size_t)needed;
    return 1;
}

static int h3_key_file(h3_key *key, const char *role, const char *path) {
    if (!path) return h3_key_append(key, "|%s=none", role);
    if (key->content_ctx) {
        const char *hash = h3_file_hash(key->content_ctx, path);
        return hash && h3_key_append(key, "|%s=%zu:%s:sha256=%s", role,
                                    strlen(path), path, hash);
    }
    struct stat status;
    if (stat(path, &status) != 0)
        return h3_key_append(key, "|%s=%zu:%s:missing", role,
                             strlen(path), path);
    return h3_key_append(key, "|%s=%zu:%s:%lld:%lld:%ld:%lld:%ld:%llu:%llu", role, strlen(path),
                         path, (long long)status.st_size,
                         (long long)status.st_mtimespec.tv_sec,
                         status.st_mtimespec.tv_nsec,
                         (long long)status.st_ctimespec.tv_sec,
                         status.st_ctimespec.tv_nsec,
                         (unsigned long long)status.st_dev,
                         (unsigned long long)status.st_ino);
}

static int h3_key_model_tree(h3_key *key,const char *path,int depth) {
    struct stat st;
    if (depth>32 || stat(path,&st)!=0) return 0;
    if (!S_ISDIR(st.st_mode)) return h3_key_file(key,"model",path);
    struct dirent **entries=NULL; int n=scandir(path,&entries,NULL,alphasort);
    if(n<0)return 0;
    int ok=1;
    for(int i=0;i<n;i++) {
        if(ok && strcmp(entries[i]->d_name,".") && strcmp(entries[i]->d_name,"..")) {
            size_t len=strlen(path)+strlen(entries[i]->d_name)+2;char *child=malloc(len);
            if(!child)ok=0;
            else {snprintf(child,len,"%s/%s",path,entries[i]->d_name);
                ok=h3_key_model_tree(key,child,depth+1);free(child);}
        }
        free(entries[i]);
    }
    free(entries);return ok;
}

/* Hash the linked executable, not this translation unit's compile timestamp:
 * an incremental rebuild may replace the sampler/DiT without rebuilding h3.o. */
static int h3_key_executable(h3_key *key) {
    char local[1024];
    char *path=local;
    uint32_t size=(uint32_t)sizeof(local);
    if(_NSGetExecutablePath(local,&size)!=0) {
        path=malloc(size);
        if(!path)return 0;
        if(_NSGetExecutablePath(path,&size)!=0) {free(path);return 0;}
    }
    int ok = h3_key_file(key, "executable", path);
    if (path != local) free(path);
    return ok;
}

static int h3_compare_strings(const void *a,const void *b) {
    return strcmp(*(const char *const *)a,*(const char *const *)b);
}
static int h3_key_environment(h3_key *key) {
    extern char **environ;size_t count=0;
    for(char **p=environ;*p;p++) if(!strncmp(*p,"H3_",3) && strncmp(*p,"H3_PROFILE=",11))count++;
    const char **items=count ? malloc(count*sizeof(*items)) : NULL;
    if(count && !items)return 0;
    size_t i=0;for(char **p=environ;*p;p++)if(!strncmp(*p,"H3_",3) && strncmp(*p,"H3_PROFILE=",11))items[i++]=*p;
    if(count>1)qsort(items,count,sizeof(*items),h3_compare_strings);
    int ok=1;for(i=0;i<count && ok;i++)ok=h3_key_append(key,"|env=%s",items[i]);
    free(items);return ok;
}

static char *h3_conditioning_key(h3_ctx *content_ctx, const char *model_identity,const char *prompt, const h3_params *params,
                                 int render_width, int render_height, int ref2va) {
    h3_key key = {.content_ctx = content_ctx};
    if (!h3_key_append(&key, "mode=%d|prompt=%zu:%s", ref2va,
                       strlen(prompt), prompt)) goto failed;
    if (!h3_key_append(&key,"|models=%s",model_identity)) goto failed;
    if (!ref2va && !params->first_frame && !params->last_frame && !params->guide_count) return key.text;
    if (!h3_key_append(&key, "|render=%dx%d|frames=%d|image-size=%d",
                       render_width, render_height, params->frames,
                       params->reference_image_size) ||
        !h3_key_file(&key, "first", params->first_frame) ||
        !h3_key_file(&key, "last", params->last_frame)) goto failed;
    for (size_t index = 0; index < params->reference_count; index++) {
        const h3_reference *reference = &params->references[index];
        if (!h3_key_append(&key, "|ref=%d:%d", reference->kind,
                           reference->include_embedded_audio) ||
            !h3_key_file(&key, "media", reference->path) ||
            !h3_key_file(&key, "audio", reference->audio_path)) goto failed;
    }
    for (size_t i = 0; i < params->guide_count; i++) {
        const h3_guide *g = &params->guides[i];
        if (!h3_key_append(&key,"|guide=%d:%d:%d:%d:%d",g->kind,g->frame_index,
                g->source_start_frame,g->source_frames,g->include_embedded_audio) ||
            !h3_key_file(&key,"guide-media",g->path) ||
            !h3_key_file(&key,"guide-audio",g->audio_path)) goto failed;
    }
    return key.text;
failed:
    free(key.text);
    return NULL;
}

static char *h3_prepared_key(const h3_ctx *ctx, const char *conditioning,
                             const h3_params *params,
                             int render_width, int render_height) {
    h3_key key = {0};
    if (!h3_key_append(
            &key,
            "%s|shape=%dx%dx%d|steps=%d|layers=%d|reuse-core=%d|reduce=%d"
            "|row-fc2=%d|reference-rope=%d|ssd-streaming=%d"
            "|slow=%d%d%d%d%d%d%d%d%d%d",
            conditioning, render_width, render_height, params->frames,
            params->steps, params->dit_layers, params->core_reuse,
            params->token_reduction, params->use_int8_row_fc2,
            params->use_reference_rope,
            params->ssd_streaming,
            params->use_slower_bf16_mlp,
            params->use_slower_bf16_qkv,
            params->use_slower_bf16_attention_output,
            params->use_slower_row_major_attention_output,
            params->use_slower_unfused_int8_inputs,
            params->use_slower_unfused_qkv_rope,
            params->use_slower_scalar_qkv_rms,
            params->use_slower_uncached_int8_scales,
            params->use_slower_dynamic_fc1_k,
            params->use_slower_grouped_quantizer)) {
        free(key.text);
        return NULL;
    }
    if ((params->sampler != H3_SAMPLER_EULER ||
         params->scheduler != H3_SCHEDULER_SERVING ||
         params->video_shift != 12.0f || params->audio_shift != 3.0f) &&
        !h3_key_append(&key, "|sampler=%d|scheduler=%d|shifts=%a,%a",
            params->sampler, params->scheduler,
            (double)params->video_shift, (double)params->audio_shift)) {
        free(key.text);
        return NULL;
    }
    if (ctx->loras && ctx->lora_target != H3_LORA_FL2VA &&
        !h3_key_append(&key, "|lora-target=%d", ctx->lora_target)) {
        free(key.text);
        return NULL;
    }
    if (ctx->loras && !h3_key_append(&key, "|loras=%s",
            h3_lora_stack_identity(ctx->loras))) {
        free(key.text);
        return NULL;
    }
    return key.text;
}

/* Adapter identity describes its immutable, already-loaded tensor snapshot. */
static int h3_exact_digest(h3_ctx *ctx, const char *prompt, const h3_params *params,
        int width, int height, int ref2va, const char *const paths[6], unsigned char digest[32]) {
    h3_key models = {.content_ctx = ctx};
    int ok = 1;
    for (size_t i = 0; i < 6 && ok; i++) ok = h3_key_model_tree(&models, paths[i], 0);
    ok = ok && h3_key_file(&models, "shader", "h3_shaders.metal") &&
        h3_key_environment(&models) && h3_key_executable(&models);
    char *conditioning = ok ? h3_conditioning_key(ctx, models.text, prompt, params,
                                                width, height, ref2va) : NULL;
    char *prepared = conditioning ? h3_prepared_key(ctx, conditioning, params, width, height) : NULL;
    h3_key identity = {0};
    ok = prepared && h3_key_append(&identity, "%s|native=%s|build=%s %s|digest=%s",
        prepared, H3_VERSION, __DATE__, __TIME__, params->request_digest ? params->request_digest : "");
    if (ok) CC_SHA256(identity.text, (CC_LONG)identity.length, digest);
    free(identity.text); free(prepared); free(conditioning); free(models.text);
    return ok;
}

static char *h3_resident_key(const h3_ctx *ctx, const h3_params *params,
                             int ref2va,const char *model_identity) {
    /* Keep one precision/adapter identity convention. Only request-dependent
     * fields are normalized; actual sequence-policy compatibility is checked
     * after the new packed layout has been built. */
    h3_params resident = *params;
    resident.frames = 0;
    resident.steps = 0;
    resident.sampler = H3_SAMPLER_EULER;
    resident.scheduler = H3_SCHEDULER_SERVING;
    resident.video_shift = 12.0f;
    resident.audio_shift = 3.0f;
    h3_key key={0};
    if(!h3_key_append(&key,"%s|%s",ref2va ? "ref2va" : "fl2va",model_identity)) {free(key.text);return NULL;}
    char *result=h3_prepared_key(ctx,key.text,&resident,0,0);free(key.text);return result;
}

static int h3_text_embedding_copy(h3_text_embedding *destination,
                                  const h3_text_embedding *source) {
    memset(destination, 0, sizeof(*destination));
    if (!source || !source->tokens || !source->width || !source->values ||
        source->tokens > SIZE_MAX / source->width) return 0;
    size_t elements = source->tokens * source->width;
    if (elements > SIZE_MAX / sizeof(*destination->values)) return 0;
    destination->values = malloc(elements * sizeof(*destination->values));
    if (source->tags)
        destination->tags = malloc(source->tokens * sizeof(*destination->tags));
    if (!destination->values || (source->tags && !destination->tags)) {
        h3_text_embedding_free(destination);
        return 0;
    }
    memcpy(destination->values, source->values,
           elements * sizeof(*destination->values));
    if (source->tags)
        memcpy(destination->tags, source->tags,
               source->tokens * sizeof(*destination->tags));
    destination->tokens = source->tokens;
    destination->width = source->width;
    destination->gpu_stats = source->gpu_stats;
    return 1;
}

static int h3_conditioning_cache_store(
        h3_ctx *ctx, const char *key, const h3_text_embedding *text,
        const float *video, size_t video_elements,
        const float *audio, size_t audio_elements,
        const h3_layout_ref *references, size_t reference_count,
        int conditioned) {
    h3_text_embedding copy;
    if (!h3_text_embedding_copy(&copy, text)) return 0;
    float *video_copy = NULL;
    float *audio_copy = NULL;
    h3_layout_ref *reference_copy = NULL;
    char *key_copy = strdup(key);
    if (video_elements) {
        video_copy = malloc(video_elements * sizeof(*video_copy));
        if (video_copy) memcpy(video_copy, video,
                               video_elements * sizeof(*video_copy));
    }
    if (audio_elements) {
        audio_copy = malloc(audio_elements * sizeof(*audio_copy));
        if (audio_copy) memcpy(audio_copy, audio,
                               audio_elements * sizeof(*audio_copy));
    }
    if (reference_count) {
        reference_copy = malloc(reference_count * sizeof(*reference_copy));
        if (reference_copy) memcpy(reference_copy, references,
                                   reference_count * sizeof(*reference_copy));
    }
    if (!key_copy || (video_elements && !video_copy) ||
        (audio_elements && !audio_copy) ||
        (reference_count && !reference_copy)) {
        free(key_copy); free(video_copy); free(audio_copy); free(reference_copy);
        h3_text_embedding_free(&copy);
        return 0;
    }
    h3_conditioning_cache_clear(ctx);
    ctx->conditioning_key = key_copy;
    ctx->conditioning_tokens = copy.tokens;
    ctx->conditioning_width = copy.width;
    ctx->conditioning_values = copy.values;
    ctx->conditioning_tags = copy.tags;
    ctx->conditioning_video_rows = video_copy;
    ctx->conditioning_video_elements = video_elements;
    ctx->conditioning_audio_rows = audio_copy;
    ctx->conditioning_audio_elements = audio_elements;
    ctx->conditioning_references = reference_copy;
    ctx->conditioning_reference_count = reference_count;
    ctx->conditioning_present = conditioned;
    return 1;
}

static int h3_conditioning_cache_load(
        const h3_ctx *ctx, h3_text_embedding *text,
        float **video, size_t *video_elements,
        float **audio, size_t *audio_elements,
        h3_layout_ref **references, size_t *reference_count,
        int *conditioned) {
    h3_text_embedding source = {
        ctx->conditioning_tokens, ctx->conditioning_width,
        ctx->conditioning_values, {0}, ctx->conditioning_tags};
    if (!h3_text_embedding_copy(text, &source)) return 0;
    *video = NULL;
    *audio = NULL;
    *references = NULL;
    *video_elements = ctx->conditioning_video_elements;
    *audio_elements = ctx->conditioning_audio_elements;
    *reference_count = ctx->conditioning_reference_count;
    *conditioned = ctx->conditioning_present;
    if (*video_elements) {
        *video = malloc(*video_elements * sizeof(**video));
        if (*video) memcpy(*video, ctx->conditioning_video_rows,
                           *video_elements * sizeof(**video));
    }
    if (*audio_elements) {
        *audio = malloc(*audio_elements * sizeof(**audio));
        if (*audio) memcpy(*audio, ctx->conditioning_audio_rows,
                           *audio_elements * sizeof(**audio));
    }
    if (*reference_count) {
        *references = malloc(*reference_count * sizeof(**references));
        if (*references) memcpy(*references, ctx->conditioning_references,
                                *reference_count * sizeof(**references));
    }
    if ((*video_elements && !*video) || (*audio_elements && !*audio) ||
        (*reference_count && !*references)) {
        h3_text_embedding_free(text);
        free(*video); free(*audio); free(*references);
        *video = NULL; *audio = NULL; *references = NULL;
        return 0;
    }
    return 1;
}

static void h3_augment_span(float *values, size_t count, uint64_t seed) {
    h3_rng rng;
    h3_rng_seed(&rng, seed);
    for (size_t index = 0; index < count; index++)
        values[index] = 0.999f * values[index] +
                        0.001f * h3_rng_normal(&rng);
}

static void h3_augment_conditions(const h3_params *params,
    const h3_layout_ref *references, const h3_layout_guide *guides, size_t guide_count,
    float *video) {
    size_t vo=0;
    for (size_t i=0;i<params->reference_count;i++) {
        const h3_layout_ref *r=&references[i];
        size_t nv=(size_t)r->latent_t*r->latent_h*r->latent_w/4*96;
        if (nv) h3_augment_span(video+vo,nv,params->seed);
        vo+=nv;
    }
    for (size_t i=0;i<guide_count;i++) {
        const h3_layout_guide *g=&guides[i];
        size_t nv=(size_t)g->latent_t*g->latent_h*g->latent_w/4*96;
        if (nv) h3_augment_span(video+g->video_offset*96,nv,params->seed);
    }
}

void h3_set_error(h3_ctx *ctx, const char *format, ...) {
    char *destination = ctx ? ctx->error : h3_global_error;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(destination, 512, format, arguments);
    va_end(arguments);
    char *code=ctx ? ctx->error_code : h3_global_error_code;
    char *field=ctx ? ctx->error_field : h3_global_error_field;
    snprintf(code,64,"engine_failed"); field[0]=0;
    static const char *codes[]={"invalid_request","guide_overlap","guide_out_of_range",
        "missing_model","profile_incompatible","latent_incompatible","checkpoint_incompatible",
        "dependency_changed","cancelled","output_failed"};
    for(size_t i=0;i<sizeof(codes)/sizeof(*codes);i++) if(strstr(destination,codes[i])) {
        snprintf(code,64,"%s",codes[i]); break;
    }
    if (!strcmp(code,"engine_failed") && (strstr(destination,"missing required model") || strstr(destination,"require the Ref2VA checkpoint")))
        snprintf(code,64,"missing_model");
    const char *guide=strstr(destination,"guides[");
    if (guide) {
        const char *end=strchr(guide,']');
        if(end) {
            end++;
            if(*end=='.') {end++;while(isalnum((unsigned char)*end) || *end=='_')end++;}
            if(end-guide<128)snprintf(field,128,"%.*s",(int)(end-guide),guide);
        }
    }
    if(!*field && strstr(destination,"params.first_frame"))snprintf(field,128,"params.first_frame");
    if(!*field && strstr(destination,"params.last_frame"))snprintf(field,128,"params.last_frame");
    if(!*field && strstr(destination,"mask"))snprintf(field,128,"params.mask_path");
    if(!*field && strstr(destination,"initial AV"))snprintf(field,128,"params.initial_latent_path");
    if(!*field && !strcmp(code,"checkpoint_incompatible"))snprintf(field,128,"params.resume_path");
}

static int h3_is_file(const char *path) {
    struct stat status;
    return stat(path, &status) == 0 && S_ISREG(status.st_mode);
}

static char *h3_path(const char *root, const char *relative) {
    size_t size = strlen(root) + strlen(relative) + 2;
    char *result = malloc(size);
    if (result) snprintf(result, size, "%s/%s", root, relative);
    return result;
}

static int h3_require_file(h3_ctx *ctx, const char *relative) {
    char *path = h3_path(ctx->model_dir, relative);
    if (!path) {
        h3_set_error(ctx, "out of memory resolving model path");
        return 0;
    }
    int exists = h3_is_file(path);
    if (!exists) h3_set_error(ctx, "missing required model file: %s", path);
    free(path);
    return exists;
}

static int h3_inventory(h3_ctx *ctx, const char *relative, int optional,
                        h3_component_info *info) {
    char *path = h3_path(ctx->model_dir, relative);
    if (!path) {
        h3_set_error(ctx, "out of memory resolving component path");
        return 0;
    }
    char detail[384];
    int ok = optional ?
        h3_st_inventory_dir_optional(path, info, detail, sizeof(detail)) :
        h3_st_inventory_dir(path, info, detail, sizeof(detail));
    if (!ok) h3_set_error(ctx, "%s", detail);
    free(path);
    return ok;
}

h3_ctx *h3_load_dir(const char *model_dir, const h3_lora_spec *loras,
                    size_t lora_count, h3_lora_model target) {
    h3_global_error[0] = '\0';
    h3_global_error_code[0]=0;h3_global_error_field[0]=0;
    if (target != H3_LORA_FL2VA && target != H3_LORA_REF2VA) {
        h3_set_error(NULL, "invalid LoRA model target");
        return NULL;
    }
    if (lora_count && !loras) {
        h3_set_error(NULL, "LoRA specifications are required");
        return NULL;
    }
    if (!model_dir || !*model_dir) {
        h3_set_error(NULL, "model directory is required");
        return NULL;
    }
    h3_ctx *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        h3_set_error(NULL, "out of memory creating H3 context");
        return NULL;
    }
    ctx->lora_target=target;
    ctx->model_dir = strdup(model_dir);
    if (!ctx->model_dir) {
        h3_set_error(NULL, "out of memory copying model path");
        free(ctx);
        return NULL;
    }
    if (!h3_require_file(ctx, "FL2VA/transformer/config.json") ||
        !h3_require_file(ctx, "FL2VA/tokenizer/tokenizer.json") ||
        !h3_inventory(ctx, "FL2VA/text_encoder", 0, &ctx->model.text_encoder) ||
        !h3_inventory(ctx, "FL2VA/transformer", 0, &ctx->model.fl2va_transformer) ||
        !h3_inventory(ctx, "FL2VA/video_vae/source", 0, &ctx->model.video_vae) ||
        !h3_inventory(ctx, "FL2VA/audio_vae", 0, &ctx->model.audio_vae)) {
        h3_set_error(NULL,"%s",ctx->error);
        h3_free(ctx);
        return NULL;
    }
    /* Optional references use the same single-file/sharded discovery as FL2VA.
     * Missing or empty directories must not prevent prompt-only generation. */
    if (!h3_inventory(ctx, "Ref2VA/transformer", 1,
                      &ctx->model.ref2va_transformer)) {
        h3_set_error(NULL,"%s",ctx->error);
        h3_free(ctx);
        return NULL;
    }
    int has_ref2va = ctx->model.ref2va_transformer.files != 0;
    int active_loras = 0;
    for (size_t i = 0; i < lora_count; i++) {
        if (!isfinite(loras[i].strength)) {
            h3_set_error(NULL, "LoRA strength must be finite");
            h3_free(ctx);
            return NULL;
        }
        active_loras |= loras[i].strength != 0.0f;
    }
    if (active_loras) {
        ctx->lora_target = target;
        if (target == H3_LORA_REF2VA && !has_ref2va) {
            h3_set_error(NULL, "active Ref2VA LoRAs require the Ref2VA checkpoint");
            h3_free(ctx);
            return NULL;
        }
        char *base_path = h3_path(model_dir, target == H3_LORA_REF2VA ?
                                  "Ref2VA/transformer" : "FL2VA/transformer");
        char detail[512] = "out of memory resolving target transformer";
        h3_weight_store *base = base_path ? h3_weight_store_open(
            base_path, detail, sizeof(detail)) : NULL;
        free(base_path);
        if (base) {
            ctx->loras = h3_lora_stack_load(loras, lora_count, base,
                                            detail, sizeof(detail));
            h3_weight_store_free(base);
        }
        if (!ctx->loras) {
            h3_set_error(NULL, "%s", detail);
            h3_free(ctx);
            return NULL;
        }
    }
    char metal_error[256];
    if (!h3_metal_probe(&ctx->device, metal_error, sizeof(metal_error))) {
        h3_set_error(ctx, "%s", metal_error);
        h3_set_error(NULL,"%s",ctx->error);
        h3_free(ctx);
        return NULL;
    }
    return ctx;
}

void h3_free(h3_ctx *ctx) {
    if (!ctx) return;
    h3_cache_clear(ctx);
    h3_lora_stack_free(ctx->loras);
    while (ctx->file_digests) {
        h3_file_digest *entry = ctx->file_digests;
        ctx->file_digests = entry->next;
        free(entry->path);
        free(entry);
    }
    free(ctx->model_dir);
    free(ctx);
}

const h3_lora_spec *h3_get_loras(const h3_ctx *ctx, size_t *count) {
    if (count) *count = 0;
    return ctx && ctx->loras ? h3_lora_stack_specs(ctx->loras, count) : NULL;
}

const char *h3_last_error(const h3_ctx *ctx) {
    return ctx ? ctx->error : h3_global_error;
}
const char *h3_last_error_code(const h3_ctx *ctx) {
    return ctx ? ctx->error_code : h3_global_error_code;
}
const char *h3_last_error_field(const h3_ctx *ctx) {
    return ctx ? ctx->error_field : h3_global_error_field;
}

const h3_device_info *h3_device(const h3_ctx *ctx) {
    return ctx ? &ctx->device : NULL;
}

const h3_model_info *h3_model(const h3_ctx *ctx) {
    return ctx ? &ctx->model : NULL;
}

static int h3_guide_video_span(const h3_guide *guide) {
    int tokens=h3_video_encoder_latent_t(guide->kind==H3_GUIDE_IMAGE ? 1 : guide->source_frames);
    /* Sum upstream FRAME_PER_TOKEN=(1,4,4,4,4) over the encoded clip. */
    return 17*(tokens/5)+(tokens%5 ? 4*(tokens%5)-3 : 0);
}

static int h3_valid_params(h3_ctx *ctx, const h3_params *params) {
    if (!params) {
        h3_set_error(ctx, "generation parameters are required");
        return 0;
    }
    if (params->width < 32 || params->height < 32 ||
        params->width % H3_CANVAS_MULTIPLE ||
        params->height % H3_CANVAS_MULTIPLE) {
        h3_set_error(ctx, "width and height must be multiples of 32 and at least 32");
        return 0;
    }
    if ((int64_t)params->width * params->height > H3_MAX_PIXELS) {
        h3_set_error(ctx, "canvas exceeds the released 768*1344 pixel limit");
        return 0;
    }
    if ((params->render_width == 0) != (params->render_height == 0)) {
        h3_set_error(ctx, "render width and height must be set together");
        return 0;
    }
    if (params->render_width) {
        if (params->render_width < 32 || params->render_height < 32 ||
            params->render_width % H3_CANVAS_MULTIPLE ||
            params->render_height % H3_CANVAS_MULTIPLE ||
            params->render_width > params->width ||
            params->render_height > params->height ||
            (int64_t)params->render_width * params->height !=
                (int64_t)params->render_height * params->width) {
            h3_set_error(ctx,
                "internal render canvas must be same-aspect multiples of 32 "
                "no larger than the output canvas");
            return 0;
        }
    }
    if (params->frames < 5 || params->frames > 362 || h3_align_frame_count(params->frames) > 362) {
        h3_set_error(ctx, "frames must align within the released 5..362 range");
        return 0;
    }
    if (params->sampler < H3_SAMPLER_EULER ||
        params->sampler > H3_SAMPLER_ER_SDE ||
        params->scheduler < H3_SCHEDULER_SERVING ||
        params->scheduler > H3_SCHEDULER_BETA57 ||
        !isfinite(params->video_shift) || params->video_shift <= 0.0f ||
        !isfinite(params->audio_shift) || params->audio_shift <= 0.0f) {
        h3_set_error(ctx, "invalid sampler, scheduler, or finite positive shifts");
        return 0;
    }
    if ((params->scheduler == H3_SCHEDULER_SERVING &&
         params->sampler != H3_SAMPLER_EULER) ||
        (params->scheduler != H3_SCHEDULER_SERVING &&
         (params->denoise_reuse != 1 || params->core_reuse != 1 ||
          params->token_reduction || params->dit_layers != H3_DEFAULT_DIT_LAYERS ||
          params->render_width || params->render_height))) {
        h3_set_error(ctx, "serving requires Euler; simple/beta/beta57 require full canvas, 50 blocks, no token reduction, reuse/core-reuse 1");
        return 0;
    }
    if (params->steps < (params->scheduler == H3_SCHEDULER_SERVING ? 2 : 1) ||
        params->steps > H3_MAX_STEPS) {
        h3_set_error(ctx, "denoising steps must be in [2, 1000] for serving or [1, 1000] for simple/beta/beta57");
        return 0;
    }
    if (params->denoise_reuse < 1 || params->denoise_reuse > 3) {
        h3_set_error(ctx, "denoise reuse must be in [1, 3]");
        return 0;
    }
    if (params->dit_layers < H3_MIN_DIT_LAYERS ||
        params->dit_layers > H3_DEFAULT_DIT_LAYERS) {
        h3_set_error(ctx, "DiT layers must be in [35, 50]");
        return 0;
    }
    if (params->core_reuse < 1 || params->core_reuse > 6) {
        h3_set_error(ctx, "core reuse must be in [1, 6]");
        return 0;
    }
    if (params->token_reduction != 0 && params->token_reduction != 1) {
        h3_set_error(ctx, "token reduction must be zero or one");
        return 0;
    }
    if (params->use_int8_row_fc2 != 0 &&
        params->use_int8_row_fc2 != 1) {
        h3_set_error(ctx, "int8 row FC2 must be zero or one");
        return 0;
    }
    if (params->use_reference_rope != 0 &&
        params->use_reference_rope != 1) {
        h3_set_error(ctx, "reference RoPE must be zero or one");
        return 0;
    }
    if (params->ssd_streaming != 0 && params->ssd_streaming != 1) {
        h3_set_error(ctx, "SSD streaming must be zero or one");
        return 0;
    }
    if (params->ssd_streaming && params->use_int8_row_fc2) {
        h3_set_error(ctx, "SSD streaming uses original BF16 weights and cannot "
                         "be combined with int8 row FC2");
        return 0;
    }
    if (params->use_int8_row_fc2 && params->use_slower_bf16_mlp) {
        h3_set_error(ctx, "int8 row FC2 cannot be combined with the BF16 MLP");
        return 0;
    }
    if (params->use_int8_row_fc2 && !h3_device(ctx)->metal4) {
        h3_set_error(ctx, "int8 row FC2 requires an M5-class Metal 4 GPU");
        return 0;
    }
    if (params->preview_denoise != 0 && params->preview_denoise != 1) {
        h3_set_error(ctx, "denoising preview must be zero or one");
        return 0;
    }
    if (params->preview_denoise && !params->on_frame) {
        h3_set_error(ctx, "denoising preview requires a frame callback");
        return 0;
    }
    if (params->core_reuse > 1 && params->denoise_reuse > 1) {
        h3_set_error(ctx, "core reuse and denoiser reuse cannot be combined");
        return 0;
    }
    if (params->reference_count && !params->references) {
        h3_set_error(ctx, "reference_count is nonzero but references is NULL");
        return 0;
    }
    if (params->reference_count > 12) {
        h3_set_error(ctx, "Ref2VA supports at most 12 references");
        return 0;
    }
    if (params->reference_image_size != H3_REFERENCE_IMAGE_MATCH &&
        params->reference_image_size != H3_REFERENCE_IMAGE_MAX) {
        h3_set_error(ctx, "unknown reference image sizing policy");
        return 0;
    }
    if ((params->guide_count && !params->guides) || params->guide_count > 64) {
        h3_set_error(ctx, "guides must contain at most 64 descriptors"); return 0;
    }
    for (size_t i=0; i<params->guide_count; i++) {
        const h3_guide *g=&params->guides[i];
        int target=h3_align_frame_count(params->frames);
        if(g->frame_index<0 || g->frame_index>=target || g->source_frames>target-g->frame_index) {
            h3_set_error(ctx,"guide_out_of_range: guides[%zu].frame_index does not fit target frames",i);return 0;
        }
        if (g->kind < H3_GUIDE_IMAGE || g->kind > H3_GUIDE_CLIP_AUDIO ||
            !g->path || !*g->path ||
            g->source_start_frame < 0 || g->source_frames < 0 ||
            ((g->kind == H3_GUIDE_CLIP || g->kind == H3_GUIDE_CLIP_AUDIO) &&
             (g->source_frames < 5 || g->source_frames % 17 != 5)) ||
            (g->kind == H3_GUIDE_IMAGE && (g->source_start_frame || g->source_frames > 1)) ||
            g->include_embedded_audio<0 || g->include_embedded_audio>1 ||
            (g->kind!=H3_GUIDE_CLIP_AUDIO &&
             (g->audio_path || g->include_embedded_audio)) ||
            (g->kind == H3_GUIDE_CLIP_AUDIO && !g->audio_path && !g->include_embedded_audio)) {
            h3_set_error(ctx,"invalid_request: guides[%zu] requires resolved placement and explicit valid source span/audio policy",i); return 0;
        }
        if(!h3_is_file(g->path) || (g->audio_path && !h3_is_file(g->audio_path))) {
            h3_set_error(ctx,"invalid_request: guides[%zu] requires readable local media files",i);return 0;
        }
        if(g->kind!=H3_GUIDE_AUDIO) {
            int span=h3_guide_video_span(g),last=h3_align_frame_count(params->frames)-1;
            if((params->first_frame && g->frame_index==0) ||
                (params->last_frame && g->frame_index<=last && g->frame_index+span>last)) {
                h3_set_error(ctx,"guide_overlap: guides[%zu] overlaps an endpoint on the video stream",i);return 0;
            }
            for(size_t j=0;j<i;j++) {
                const h3_guide *previous=&params->guides[j];
                if(previous->kind!=H3_GUIDE_AUDIO &&
                    g->frame_index<previous->frame_index+h3_guide_video_span(previous) &&
                    previous->frame_index<g->frame_index+span) {
                    h3_set_error(ctx,"guide_overlap: guides[%zu] overlaps guides[%zu] on the video stream",i,j);return 0;
                }
            }
        }
    }
    if (params->start_transition < 0 || params->start_transition > params->steps ||
        (params->start_transition && !params->initial_latent_path && !params->resume_path) ||
        (params->mask_path && !params->initial_latent_path) ||
        (params->resume_path && (params->initial_latent_path || params->mask_path))) {
        h3_set_error(ctx,"latent_incompatible: invalid initial latent, mask, start transition or resume combination"); return 0;
    }
    if ((params->initial_latent_path || params->resume_path || params->checkpoint_path || params->on_transition) &&
        (params->denoise_reuse != 1 || params->core_reuse != 1 || params->token_reduction || params->ssd_streaming ||
         params->dit_layers != H3_DEFAULT_DIT_LAYERS)) {
        h3_set_error(ctx,"profile_incompatible: stateful sampling requires uncached exact execution"); return 0;
    }
    const char *outputs[]={params->output_path,params->latent_output_path,params->checkpoint_path};
    for(size_t i=0;i<3;i++)for(size_t j=0;j<i;j++)
        if(outputs[i] && *outputs[i] && outputs[j] && !strcmp(outputs[i],outputs[j])) {
            h3_set_error(ctx,"invalid_request: media, latent and checkpoint outputs require distinct paths");return 0;
        }
    size_t images = 0, videos = 0, audio_inputs = 0;
    for (size_t index = 0; index < params->reference_count; index++) {
        const h3_reference *reference = &params->references[index];
        if (!reference->path || !*reference->path) {
            h3_set_error(ctx, "reference %zu has no input path", index + 1);
            return 0;
        }
        switch (reference->kind) {
        case H3_REFERENCE_IMAGE:
            images++;
            break;
        case H3_REFERENCE_VIDEO:
            videos++;
            if (reference->include_embedded_audio) audio_inputs++;
            break;
        case H3_REFERENCE_AUDIO:
            audio_inputs++;
            break;
        case H3_REFERENCE_VIDEO_AUDIO:
            videos++; audio_inputs++;
            if (!reference->audio_path || !*reference->audio_path) {
                h3_set_error(ctx,
                    "video+audio reference %zu has no soundtrack path",
                    index + 1);
                return 0;
            }
            break;
        default:
            h3_set_error(ctx, "reference %zu has an unknown type", index + 1);
            return 0;
        }
    }
    if (images > 9 || videos > 3 || audio_inputs > 3) {
        h3_set_error(ctx,
            "Ref2VA limits are 9 images, 3 videos, and 3 audio inputs");
        return 0;
    }
    return 1;
}

typedef struct {
    h3_ctx *ctx;
    const h3_params *params;
    int cancelled;
} h3_generation_progress;

typedef struct { const h3_params *params; int total, action; } h3_transition_control;
static int h3_transition_observe(int completed,const float *video,const float *audio,
    void *opaque,char *error,size_t error_size) {
    (void)video;(void)audio;(void)error;(void)error_size;
    h3_transition_control *control=opaque;
    control->action=control->params->on_transition ?
        control->params->on_transition(completed,control->total,control->params->callback_opaque) : 0;
    return control->action==0;
}

static void h3_progress_emit(h3_generation_progress *state, const char *phase,
                             int completed, int total) {
    if (!state || state->cancelled || !state->params->on_progress) return;
    if (state->params->on_progress(phase, completed, total,
                                   state->params->callback_opaque)) {
        state->cancelled = 1;
        h3_set_error(state->ctx, "generation cancelled during %s", phase);
    }
}

static void h3_text_progress_bridge(int completed, int total, void *opaque) {
    h3_progress_emit(opaque, "text encoder", completed, total);
}

static void h3_dit_progress_bridge(const char *phase, int completed, int total,
                                   void *opaque) {
    h3_progress_emit(opaque, phase, completed, total);
}

static void h3_vae_progress_bridge(int completed, int total, void *opaque) {
    h3_progress_emit(opaque, "video VAE load", completed, total);
}

static void h3_preview_vae_progress_bridge(int completed, int total,
                                           void *opaque) {
    h3_progress_emit(opaque, "preview VAE load", completed, total);
}

static void h3_audio_vae_progress_bridge(int completed, int total,
                                         void *opaque) {
    h3_progress_emit(opaque, "audio VAE", completed, total);
}

static void h3_audio_encoder_progress_bridge(int completed, int total,
                                             void *opaque) {
    h3_progress_emit(opaque, "audio VAE encoder", completed, total);
}

static void h3_video_encoder_progress_bridge(int completed, int total,
                                             void *opaque) {
    h3_progress_emit(opaque, "video VAE encoder", completed, total);
}

static h3_video_vae_decoder *h3_acquire_video_decoder(
        h3_ctx *ctx, const char *key, const char *weight_directory,
        int latent_height, int latent_width, h3_video_vae_progress progress,
        void *progress_opaque, int *cached, char *error, size_t error_size) {
    *cached = 0;
    if (ctx->cache_enabled && ctx->video_decoder &&
        ctx->video_decoder_key && !strcmp(ctx->video_decoder_key, key)) {
        *cached = 1;
        fprintf(stderr, "h3: video VAE cache hit\n");
        return ctx->video_decoder;
    }
    if (ctx->cache_enabled) {
        h3_video_vae_decoder_free(ctx->video_decoder);
        ctx->video_decoder = NULL;
        free(ctx->video_decoder_key);
        ctx->video_decoder_key = NULL;
    }
    h3_video_vae_decoder *decoder = h3_video_vae_decoder_load(
        weight_directory, "h3_shaders.metal", latent_height, latent_width,
        progress, progress_opaque, error, error_size);
    if (!decoder || !ctx->cache_enabled) return decoder;
    char *key_copy = strdup(key);
    if (!key_copy) {
        fprintf(stderr, "h3: warning: could not retain video VAE cache key\n");
        return decoder;
    }
    ctx->video_decoder = decoder;
    ctx->video_decoder_key = key_copy;
    *cached = 1;
    fprintf(stderr, "h3: video VAE cache miss; decoder retained\n");
    return decoder;
}

static void h3_vision_progress_bridge(int completed, int total, void *opaque) {
    h3_progress_emit(opaque, "Qwen vision", completed, total);
}

static uint8_t *h3_rgb_f32_to_u8(const float *rgb, size_t count) {
    uint8_t *output = malloc(count);
    if (!output) return NULL;
    for (size_t index = 0; index < count; index++) {
        float scaled = rgb[index] * 255.0f;
        if (scaled < 0.0f) scaled = 0.0f;
        if (scaled > 255.0f) scaled = 255.0f;
        output[index] = (uint8_t)lrintf(scaled);
    }
    return output;
}

typedef struct {
    h3_generation_progress *progress;
    h3_video_vae_decoder *decoder;
    int latent_t;
    int latent_h;
    int latent_w;
    int output_frames;
    int output_width;
    int output_height;
    int failed;
} h3_live_preview;

static int h3_deliver_denoise_preview(int completed_steps, int total_steps,
                                      const float *video_latent,
                                      size_t video_elements, void *opaque) {
    h3_live_preview *preview = opaque;
    if (!preview || !preview->progress || !preview->decoder || !video_latent) {
        if (preview && preview->progress)
            h3_set_error(preview->progress->ctx,
                         "invalid denoising preview latent");
        if (preview) preview->failed = 1;
        return 1;
    }
    size_t expected = (size_t)24 * (size_t)preview->latent_t *
                      (size_t)preview->latent_h * (size_t)preview->latent_w;
    if (video_elements != expected) {
        h3_set_error(preview->progress->ctx,
                     "invalid denoising preview latent size");
        preview->failed = 1;
        return 1;
    }
    char detail[512];
    h3_video_frames decoded;
    memset(&decoded, 0, sizeof(decoded));
    int frame_index = 0;
    if (!h3_video_vae_decoder_preview(
            preview->decoder, video_latent, preview->latent_t,
            &decoded, &frame_index, detail, sizeof(detail))) {
        h3_set_error(preview->progress->ctx,
                     "cannot decode denoising preview: %s", detail);
        preview->failed = 1;
        return 1;
    }
    size_t count = (size_t)decoded.width * (size_t)decoded.height * 3;
    uint8_t *rgb = h3_rgb_f32_to_u8(decoded.rgb, count);
    h3_video_frames_free(&decoded);
    if (!rgb) {
        h3_set_error(preview->progress->ctx,
                     "out of memory converting denoising preview");
        preview->failed = 1;
        return 1;
    }
    int source_width = preview->latent_w * H3_VAE_SPATIAL_RATIO;
    int source_height = preview->latent_h * H3_VAE_SPATIAL_RATIO;
    if (source_width != preview->output_width ||
        source_height != preview->output_height) {
        uint8_t *resized = NULL;
        if (!h3_resize_rgb24_high_quality(
                rgb, 1, source_width, source_height,
                preview->output_width, preview->output_height, &resized)) {
            free(rgb);
            h3_set_error(preview->progress->ctx,
                         "cannot resize denoising preview");
            preview->failed = 1;
            return 1;
        }
        free(rgb);
        rgb = resized;
    }
    h3_frame frame = {
        preview->output_width, preview->output_height,
        preview->output_width * 3, rgb, frame_index,
        preview->output_frames, completed_steps - 1, total_steps
    };
    int cancelled = preview->progress->params->on_frame(
        &frame, preview->progress->params->callback_opaque);
    free(rgb);
    if (cancelled) {
        h3_set_error(preview->progress->ctx,
                     "generation cancelled during denoising preview %d",
                     completed_steps);
        preview->failed = 1;
        return 1;
    }
    return 0;
}

/* Qwen consumes reference video as time-major two-frame blocks, while the
 * visual VAE and media boundary retain channel-major [3,T,H,W]. */
static float *h3_extract_vision_pair(const float *pixels, int frames,
                                     int height, int width,
                                     int first, int second) {
    if (!pixels || frames < 1 || height < 1 || width < 1 || first < 0 ||
        second < 0 || first >= frames || second >= frames) return NULL;
    size_t area = (size_t)height * (size_t)width;
    if (area > SIZE_MAX / 6 || 6 * area > SIZE_MAX / sizeof(float)) return NULL;
    float *pair = malloc(6 * area * sizeof(*pair));
    if (!pair) return NULL;
    const int times[2] = {first, second};
    for (int time = 0; time < 2; time++)
        for (int channel = 0; channel < 3; channel++) {
            size_t source = ((size_t)channel * (size_t)frames +
                             (size_t)times[time]) * area;
            size_t destination = ((size_t)time * 3 +
                                  (size_t)channel) * area;
            memcpy(pair + destination, pixels + source,
                   area * sizeof(*pair));
        }
    return pair;
}

static int h3_encode_guide(const h3_guide *g, int width, int height, int target_frames,
    const char *video_path, const char *audio_path, float *prepared_pixels,
    h3_image_fit fit, h3_layout_guide *out,
    float **video, size_t *nv, float **audio, size_t *na,
    char *error, size_t error_size) {
    memset(out,0,sizeof(*out));
    out->frame_index=g->frame_index; out->frame_count=1;
    out->video_offset=*nv/96; out->audio_offset=*na/32;
    if (g->kind != H3_GUIDE_AUDIO) {
        float *pixels=prepared_pixels; int frames=1;
        if (g->kind == H3_GUIDE_IMAGE) {
            if (!pixels && !h3_ffmpeg_read_image_f32(g->path,width,height,fit,
                    &pixels,error,error_size)) return 0;
        } else {
            frames=g->source_frames;
            if (!h3_ffmpeg_read_video_range_f32(g->path,width,height,g->source_start_frame,
                    frames,&pixels,error,error_size)) return 0;
        }
        if (g->frame_index < 0 || frames > target_frames-g->frame_index) {
            free(pixels); snprintf(error,error_size,"guide_out_of_range: visual span"); return 0;
        }
        h3_video_latent z={0};
        int ok=h3_video_vae_encode(video_path,"h3_shaders.metal",pixels,frames,height,width,
            NULL,NULL,&z,error,error_size); free(pixels); if (!ok) return 0;
        size_t n=(size_t)24*z.time*z.height*z.width;
        if (*nv > SIZE_MAX/sizeof(float)-n) { h3_video_latent_free(&z); return 0; }
        float *p=realloc(*video,(*nv+n)*sizeof(float));
        if (!p) { h3_video_latent_free(&z); return 0; } *video=p;
        ok=h3_dit_patchify_video(z.values,24,z.time,z.height,z.width,p+*nv,n);
        out->latent_t=z.time; out->latent_h=z.height; out->latent_w=z.width;
        out->frame_count=frames; *nv+=n; h3_video_latent_free(&z); if (!ok) return 0;
    }
    if (g->kind == H3_GUIDE_AUDIO || g->kind == H3_GUIDE_CLIP_AUDIO) {
        const char *path=g->audio_path ? g->audio_path : g->path;
        int max_t=(int)floor(h3_temporal(target_frames).audio_t-g->frame_index*40.0/24.0);
        int limit=g->source_frames ? (int)llround(g->source_frames*32000.0/24.0) : max_t*800;
        if(limit>max_t*800)limit=max_t*800;
        int64_t skip=(int64_t)llround(g->source_start_frame*32000.0/24.0);
        if(limit<1) {snprintf(error,error_size,"guide_out_of_range: empty remaining audio");return 0;}
        float *pcm=NULL; int samples=0;
        if (!h3_ffmpeg_read_audio_range_f32(path,skip,limit,g->source_frames>0,
                &pcm,&samples,error,error_size)) return 0;
        h3_audio_latent z={0}; int ok=h3_audio_vae_encode(audio_path,"h3_shaders.metal",
            pcm,samples,NULL,NULL,&z,error,error_size); free(pcm); if (!ok) return 0;
        if (z.length > (int)floor(h3_temporal(target_frames).audio_t-g->frame_index*40.0/24.0)) {
            h3_audio_latent_free(&z); snprintf(error,error_size,"guide_out_of_range: padded audio span"); return 0;
        }
        size_t n=(size_t)64*z.length;
        if (*na > SIZE_MAX/sizeof(float)-n) { h3_audio_latent_free(&z); return 0; }
        float *p=realloc(*audio,(*na+n)*sizeof(float));
        if (!p) { h3_audio_latent_free(&z); return 0; } *audio=p;
        for (int s=0;s<2;s++) for(int t=0;t<z.length;t++) for(int c=0;c<32;c++)
            p[*na+((size_t)s*z.length+t)*32+c]=z.values[((size_t)c*2+s)*z.length+t];
        *na+=n; out->audio_t=z.length; h3_audio_latent_free(&z);
    }
    return 1;
}

h3_result *h3_generate(h3_ctx *ctx, const char *prompt,
                       const h3_params *params) {
    if (!ctx) return NULL;
    ctx->error[0] = '\0';
    if (!prompt || !*prompt) {
        h3_set_error(ctx, "invalid_request: prompt must not be empty");
        return NULL;
    }
    if (!h3_valid_params(ctx, params)) {
        if(!strcmp(ctx->error_code,"engine_failed"))snprintf(ctx->error_code,sizeof(ctx->error_code),"invalid_request");
        return NULL;
    }
    h3_sigma_schedule sigmas;
    int schedule_ok=params->scheduler==H3_SCHEDULER_SERVING ?
        h3_serving_schedule_build(params->steps,params->video_shift,params->audio_shift,&sigmas) :
        params->scheduler==H3_SCHEDULER_SIMPLE ?
        h3_schedule_build(params->steps,params->video_shift,params->audio_shift,&sigmas) :
        h3_beta_schedule_build(params->steps,params->video_shift,params->audio_shift,
                               params->scheduler==H3_SCHEDULER_BETA57,&sigmas);
    if(!schedule_ok || params->start_transition>sigmas.steps) {
        h3_set_error(ctx,"invalid_request: cannot construct schedule or start exceeds effective transitions");
        return NULL;
    }
    /* Comfy noises the input before ER-SDE applies its first-sigma offset.
     * In particular full-noise starts must remain full noise, not shrink. */
    float initial_video_sigma=sigmas.video[params->start_transition];
    float initial_audio_sigma=sigmas.audio[params->start_transition];
    if(!h3_sampler_schedule_prepare(params->sampler,params->video_shift,params->audio_shift,&sigmas)) {
        h3_set_error(ctx,"invalid_request: cannot prepare sampler sigma schedule");return NULL;
    }
    int decode_only=params->initial_latent_path && params->start_transition==sigmas.steps;
    int ref2va = params->reference_count != 0;
    if (!decode_only && ctx->loras && (params->ssd_streaming ||
        (ref2va != (ctx->lora_target == H3_LORA_REF2VA)))) {
        h3_set_error(ctx,
            "profile_incompatible: active LoRA target must match request mode and cannot use SSD streaming");
        return NULL;
    }
    int render_width = params->render_width ? params->render_width :
                                               params->width;
    int render_height = params->render_height ? params->render_height :
                                                 params->height;
    if (!decode_only && h3_align_frame_count(params->frames) < 22) {
        h3_set_error(ctx,
            "generation requires at least one trained 22-frame decoder chunk");
        return NULL;
    }
    if (!decode_only && ref2va && !ctx->model.ref2va_transformer.files) {
        h3_set_error(ctx, "missing_model: request requires the Ref2VA checkpoint");
        return NULL;
    }
    h3_generation_progress progress = {ctx, params, 0};
    h3_temporal_shape temporal = h3_temporal(params->frames);
    int latent_w, latent_h;
    h3_latent_canvas(render_width, render_height, &latent_w, &latent_h);
    h3_tokenizer *tokenizer = NULL;
    uint32_t *ids = NULL;
    size_t token_count = 0;
    size_t visual_capacity = ref2va ? params->reference_count :
        (size_t)(params->first_frame != NULL) +
        (size_t)(params->last_frame != NULL);
    size_t visual_count = 0;
    float **condition_pixels = NULL;
    int *condition_widths = NULL;
    int *condition_heights = NULL;
    int *condition_frames = NULL;
    size_t *visual_reference_indices = NULL;
    size_t *reference_visual_indices = NULL;
    h3_vision_output *vision_outputs = NULL;
    size_t vision_output_count = 0;
    h3_reference_presentation *presentations = NULL;
    double **presentation_timestamps = NULL;
    h3_layout_ref *layout_references = NULL;
    h3_layout_guide *layout_guides = NULL;
    size_t layout_guide_count = params->guide_count +
        (params->first_frame != NULL) + (params->last_frame != NULL);
    float *condition_video_rows = NULL;
    size_t condition_video_elements = 0;
    float *condition_audio_rows = NULL;
    size_t condition_audio_elements = 0;
    h3_text_embedding text;
    memset(&text, 0, sizeof(text));
    h3_layout layout;
    memset(&layout, 0, sizeof(layout));
    h3_dit *dit = NULL;
    h3_video_vae_decoder *preview_decoder = NULL;
    h3_live_preview live_preview;
    memset(&live_preview, 0, sizeof(live_preview));
    float *video = NULL, *audio = NULL;
    h3_sampler_state *sampler_state = NULL;
    h3_latent initial = {0}, masks = {0};
    h3_video_frames frames;
    memset(&frames, 0, sizeof(frames));
    h3_audio_waveform waveform;
    memset(&waveform, 0, sizeof(waveform));
    uint8_t *rgb8 = NULL;
    h3_result *result = NULL;
    char *conditioning_key = NULL;
    char *prepared_key = NULL;
    char *resident_key = NULL;
    char *decoder_key = NULL;
    char *model_identity=NULL;
    int conditioning_hit = 0;
    int conditioned = 0;
    int dit_is_cached = 0;
    int decoder_is_cached = 0;
    char *tokenizer_path = h3_path(ctx->model_dir, ref2va ?
        "Ref2VA/tokenizer/tokenizer.json" : "FL2VA/tokenizer/tokenizer.json");
    char *text_path = h3_path(ctx->model_dir, ref2va ?
        "Ref2VA/text_encoder" : "FL2VA/text_encoder");
    char *dit_path = h3_path(ctx->model_dir, ref2va ?
        "Ref2VA/transformer" : "FL2VA/transformer");
    char *vae_path = h3_path(ctx->model_dir, ref2va ?
        "Ref2VA/video_vae/source" : "FL2VA/video_vae/source");
    char *audio_vae_path = h3_path(ctx->model_dir, ref2va ?
        "Ref2VA/audio_vae" : "FL2VA/audio_vae");
    /* Preserve source/../ exactly: source may be a symlink, so lexical
     * normalization would fingerprint a different file than the VAE reads. */
    char *vae_config_path = vae_path ? h3_path(vae_path,"../config.json") : NULL;
    if (!tokenizer_path || !text_path || !dit_path || !vae_path ||
        !audio_vae_path || !vae_config_path) {
        h3_set_error(ctx, "out of memory resolving generation model paths");
        goto cleanup;
    }
    char detail[512]={0};
    size_t expected_video=(size_t)24*temporal.video_t*latent_h*latent_w;
    size_t expected_audio=(size_t)64*temporal.audio_t;
    int controlled=params->initial_latent_path || params->resume_path ||
        params->checkpoint_path || params->on_transition;
    h3_sampler_config config={0};
    const char *dependency_paths[6]={tokenizer_path,text_path,dit_path,vae_path,audio_vae_path,vae_config_path};
    uint64_t dependency_revision=0;
    if (params->resume_path) {
        struct stat checkpoint;
        if (stat(params->resume_path,&checkpoint) || !S_ISREG(checkpoint.st_mode) || checkpoint.st_size < 136) {
            h3_set_error(ctx,"checkpoint_incompatible: missing or truncated sampler state");goto cleanup;
        }
    }
    if (params->initial_latent_path &&
        (!h3_latent_load(params->initial_latent_path,&initial,detail,sizeof(detail)) ||
         initial.video_count!=expected_video || initial.audio_count!=expected_audio ||
         initial.video_frames!=(uint64_t)temporal.video_t || initial.height!=(uint64_t)latent_h ||
         initial.width!=(uint64_t)latent_w || initial.audio_frames!=(uint64_t)temporal.audio_t)) {
        h3_set_error(ctx,"latent_incompatible: initial AV shape: %s",detail);goto cleanup;
    }
    if (params->mask_path &&
        (!h3_latent_load(params->mask_path,&masks,detail,sizeof(detail)) ||
         (masks.video_count && (masks.video_count!=expected_video || masks.video_frames!=(uint64_t)temporal.video_t ||
             masks.height!=(uint64_t)latent_h || masks.width!=(uint64_t)latent_w)) ||
         (masks.audio_count && (masks.audio_count!=expected_audio || masks.audio_frames!=(uint64_t)temporal.audio_t)))) {
        h3_set_error(ctx,"latent_incompatible: mask shape: %s",detail);goto cleanup;
    }
    for(size_t i=0;i<masks.video_count;i++) if(masks.video[i]<0 || masks.video[i]>1) {
        h3_set_error(ctx,"latent_incompatible: video mask must be in [0,1]");goto cleanup;
    }
    for(size_t i=0;i<masks.audio_count;i++) if(masks.audio[i]<0 || masks.audio[i]>1) {
        h3_set_error(ctx,"latent_incompatible: audio mask must be in [0,1]");goto cleanup;
    }
    if (decode_only) {
        if(params->latent_output_path && !h3_latent_save(params->latent_output_path,&initial,detail,sizeof(detail))) {
            h3_set_error(ctx,"output_failed: latent: %s",detail);goto cleanup;
        }
        result=h3_decode_latent_output(ctx,params->initial_latent_path,0,params,
            params->width,params->height);goto cleanup;
    }
    if (controlled) {
        config.kind=params->sampler; config.schedule=sigmas;
        config.video_shift=params->video_shift; config.audio_shift=params->audio_shift;
        config.seed=params->seed; config.start_step=params->start_transition;
        config.video_count=expected_video; config.audio_count=expected_audio;
        if (!h3_exact_digest(ctx,prompt,params,render_width,render_height,ref2va,
                             dependency_paths,config.request_digest)) {
            h3_set_error(ctx,"dependency_changed: cannot fingerprint exact checkpoint dependencies");goto cleanup;
        }
        dependency_revision=ctx->file_digest_revision;
        if (params->resume_path) {
            sampler_state=h3_sampler_state_load(params->resume_path,&config,detail,sizeof(detail));
            if (!sampler_state) {h3_set_error(ctx,"checkpoint_incompatible: %s",detail);goto cleanup;}
        }
    }
    h3_key models={.content_ctx=controlled ? ctx : NULL};
    int models_ok=1;
    if (controlled) {
        for (size_t i=0;i<6 && models_ok;i++)
            models_ok=h3_key_model_tree(&models,dependency_paths[i],0);
    } else models_ok=h3_key_model_tree(&models,ctx->model_dir,0);
    if(!models_ok ||
        !h3_key_file(&models,"shader","h3_shaders.metal") || !h3_key_environment(&models)) {
        free(models.text);h3_set_error(ctx,"out of memory fingerprinting model execution dependencies");goto cleanup;
    }
    model_identity=models.text;
    conditioning_key = h3_conditioning_key(
        controlled ? ctx : NULL, model_identity, prompt, params, render_width, render_height, ref2va);
    if (!conditioning_key) {
        h3_set_error(ctx, "out of memory constructing conditioning cache key");
        goto cleanup;
    }
    prepared_key = h3_prepared_key(
        ctx, conditioning_key, params, render_width, render_height);
    if (!prepared_key) {
        h3_set_error(ctx, "out of memory constructing prepared-model cache key");
        goto cleanup;
    }
    if (ctx->cache_enabled) {
        resident_key = h3_resident_key(ctx, params, ref2va,model_identity);
        if (!resident_key) {
            h3_set_error(ctx, "out of memory constructing resident-model cache key");
            goto cleanup;
        }
    }
    const char *vae_precision = getenv("H3_VAE_PRECISION");
    if (!vae_precision) vae_precision = "f32";
    if (strcmp(vae_precision, "f32") && strcmp(vae_precision, "bf16")) {
        h3_set_error(ctx, "H3_VAE_PRECISION must be f32 or bf16");
        goto cleanup;
    }
    h3_key decoder_cache_key = {0};
    if (!h3_key_append(&decoder_cache_key, "%s|%dx%d|%s|%s", vae_path,
                       latent_h, latent_w, vae_precision,model_identity)) {
        free(decoder_cache_key.text);
        h3_set_error(ctx, "out of memory constructing decoder cache key");
        goto cleanup;
    }
    decoder_key = decoder_cache_key.text;
    if (ctx->cache_enabled && ctx->video_decoder &&
        (!ctx->video_decoder_key || strcmp(ctx->video_decoder_key, decoder_key))) {
        h3_video_vae_decoder_free(ctx->video_decoder);
        ctx->video_decoder = NULL;
        free(ctx->video_decoder_key);
        ctx->video_decoder_key = NULL;
    }
    if (ctx->cache_enabled && ctx->dit &&
        (!ctx->dit_model_key || strcmp(ctx->dit_model_key, resident_key)))
        h3_prepared_cache_clear(ctx);
    conditioning_hit = ctx->cache_enabled && ctx->conditioning_key &&
        !strcmp(ctx->conditioning_key, conditioning_key);
    if (conditioning_hit) {
        size_t cached_reference_count = 0;
        if (!h3_conditioning_cache_load(
                ctx, &text, &condition_video_rows, &condition_video_elements,
                &condition_audio_rows, &condition_audio_elements,
                &layout_references, &cached_reference_count, &conditioned) ||
            cached_reference_count != (ref2va ? params->reference_count : 0)) {
            h3_set_error(ctx, "cannot restore cached conditioning");
            goto cleanup;
        }
        if (ctx->conditioning_guide_count != layout_guide_count) {
            h3_set_error(ctx,"cached guide descriptor count mismatch"); goto cleanup;
        }
        if (layout_guide_count) {
            layout_guides=malloc(layout_guide_count*sizeof(*layout_guides));
            if (!layout_guides) { h3_set_error(ctx,"out of memory restoring guides"); goto cleanup; }
            memcpy(layout_guides,ctx->conditioning_guides,layout_guide_count*sizeof(*layout_guides));
        }
        fprintf(stderr, "h3: conditioning cache hit\n");
    } else {
    if (visual_capacity) {
        condition_pixels = calloc(visual_capacity, sizeof(*condition_pixels));
        condition_widths = calloc(visual_capacity, sizeof(*condition_widths));
        condition_heights = calloc(visual_capacity, sizeof(*condition_heights));
        condition_frames = calloc(visual_capacity, sizeof(*condition_frames));
        visual_reference_indices = calloc(
            visual_capacity, sizeof(*visual_reference_indices));
        reference_visual_indices = malloc(
            params->reference_count * sizeof(*reference_visual_indices));
        if (reference_visual_indices)
            for (size_t index = 0; index < params->reference_count; index++)
                reference_visual_indices[index] = SIZE_MAX;
        if (ref2va) {
            layout_references = calloc(visual_capacity,
                                       sizeof(*layout_references));
            presentations = calloc(params->reference_count,
                                   sizeof(*presentations));
            presentation_timestamps = calloc(
                params->reference_count, sizeof(*presentation_timestamps));
        }
        if (!condition_pixels || !condition_widths || !condition_heights ||
            !condition_frames || !visual_reference_indices ||
            !reference_visual_indices ||
            (ref2va && (!layout_references || !presentations ||
                        !presentation_timestamps))) {
            h3_set_error(ctx, "out of memory preparing visual references");
            goto cleanup;
        }
    }
    h3_progress_emit(&progress, "tokenizer", 0, 1);
    tokenizer = h3_tokenizer_load(tokenizer_path, detail, sizeof(detail));
    if (!tokenizer) {
        h3_set_error(ctx, "%s", detail);
        goto cleanup;
    }
    h3_progress_emit(&progress, "tokenizer", 1, 1);
    if (progress.cancelled) goto cleanup;
    if (ref2va) {
        for (size_t index = 0; index < params->reference_count; index++) {
            const h3_reference *reference = &params->references[index];
            if (reference->kind == H3_REFERENCE_AUDIO) {
                presentations[index].kind = H3_PRESENTATION_AUDIO;
                layout_references[index] = (h3_layout_ref){
                    H3_LAYOUT_REF_AUDIO, 0, 0, 0, 0};
                continue;
            }
            int source_width, source_height, media_width, media_height;
            if (!h3_ffprobe_visual_size(reference->path,
                                        &source_width, &source_height,
                                        detail, sizeof(detail))) {
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            if (reference->kind == H3_REFERENCE_IMAGE) {
                if (!h3_reference_image_canvas(
                        source_width, source_height,
                        render_width, render_height,
                        params->reference_image_size == H3_REFERENCE_IMAGE_MAX ?
                        2048 : 0, &media_width, &media_height)) {
                    h3_set_error(ctx,
                        "cannot resolve reference image %zu canvas", index + 1);
                    goto cleanup;
                }
                if (!h3_ffmpeg_read_image_f32(
                        reference->path, media_width, media_height,
                        H3_IMAGE_FIT_STRETCH, &condition_pixels[visual_count],
                        detail, sizeof(detail))) {
                    h3_set_error(ctx, "%s", detail);
                    goto cleanup;
                }
                condition_frames[visual_count] = 1;
                presentations[index].kind = H3_PRESENTATION_IMAGE;
                presentations[index].vision_count = 1;
                vision_output_count++;
            } else {
                if (!h3_reference_video_canvas(
                        source_width, source_height,
                        &media_width, &media_height)) {
                    h3_set_error(ctx,
                        "cannot resolve reference video %zu canvas", index + 1);
                    goto cleanup;
                }
                if (!h3_ffmpeg_read_video_f32(
                        reference->path, media_width, media_height,
                        temporal.frame_count, &condition_pixels[visual_count],
                        &condition_frames[visual_count],
                        detail, sizeof(detail))) {
                    h3_set_error(ctx, "%s", detail);
                    goto cleanup;
                }
                size_t samples = ((size_t)condition_frames[visual_count] + 11) / 12;
                size_t blocks = (samples + 1) / 2;
                if (vision_output_count > SIZE_MAX - blocks) {
                    h3_set_error(ctx, "reference video vision count overflows");
                    goto cleanup;
                }
                presentation_timestamps[index] = malloc(
                    blocks * sizeof(*presentation_timestamps[index]));
                if (!presentation_timestamps[index]) {
                    h3_set_error(ctx,
                        "out of memory allocating reference video timestamps");
                    goto cleanup;
                }
                for (size_t block = 0; block < blocks; block++) {
                    size_t first = 2 * block;
                    size_t second = first + 1 < samples ? first + 1 : first;
                    presentation_timestamps[index][block] =
                        ((double)first + (double)second) / 4.0;
                }
                presentations[index].kind = H3_PRESENTATION_VIDEO;
                presentations[index].vision_count = blocks;
                presentations[index].timestamps = presentation_timestamps[index];
                vision_output_count += blocks;
            }
            condition_widths[visual_count] = media_width;
            condition_heights[visual_count] = media_height;
            visual_reference_indices[visual_count] = index;
            reference_visual_indices[index] = visual_count;
            int ref_latent_w, ref_latent_h;
            h3_latent_canvas(media_width, media_height,
                             &ref_latent_w, &ref_latent_h);
            layout_references[index] = (h3_layout_ref){
                reference->kind == H3_REFERENCE_IMAGE ?
                    H3_LAYOUT_REF_IMAGE : H3_LAYOUT_REF_VIDEO,
                h3_video_encoder_latent_t(condition_frames[visual_count]),
                ref_latent_h, ref_latent_w, 0};
            visual_count++;
        }

        size_t total_audio_samples = 0;
        for (size_t index = 0; index < params->reference_count; index++) {
            const h3_reference *reference = &params->references[index];
            const char *audio_path = NULL;
            int truncate = 0;
            int max_samples = 32000 * 15;
            if (reference->kind == H3_REFERENCE_AUDIO) {
                audio_path = reference->path;
            } else if (reference->kind == H3_REFERENCE_VIDEO_AUDIO) {
                audio_path = reference->audio_path;
                truncate = 1;
            } else if (reference->kind == H3_REFERENCE_VIDEO &&
                       reference->include_embedded_audio) {
                audio_path = reference->path;
                truncate = 1;
            }
            if (!audio_path) continue;
            if (truncate) {
                size_t visual = reference_visual_indices[index];
                if (visual == SIZE_MAX) {
                    h3_set_error(ctx,
                        "video soundtrack %zu has no decoded video", index + 1);
                    goto cleanup;
                }
                max_samples = (int)llround(
                    (double)condition_frames[visual] * 32000.0 / H3_FPS);
                if (max_samples < 64000) {
                    h3_set_error(ctx,
                        "video soundtrack %zu requires at least 2 seconds; "
                        "request at least 56 output frames", index + 1);
                    goto cleanup;
                }
            }
            float *pcm = NULL;
            int samples = 0;
            if (!h3_ffmpeg_read_audio_f32(
                    audio_path, max_samples, truncate, &pcm, &samples,
                    detail, sizeof(detail))) {
                free(pcm);
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            if (samples < 64000) {
                free(pcm); h3_set_error(ctx,"reference audio requires at least 2 seconds at 32 kHz"); goto cleanup;
            }
            if ((size_t)samples > (size_t)32000 * 15 - total_audio_samples) {
                free(pcm);
                h3_set_error(ctx,
                    "ordered reference audio exceeds 15 seconds in total");
                goto cleanup;
            }
            h3_audio_latent latent;
            memset(&latent, 0, sizeof(latent));
            if (!h3_audio_vae_encode(
                    audio_vae_path, "h3_shaders.metal", pcm, samples,
                    h3_audio_encoder_progress_bridge, &progress, &latent,
                    detail, sizeof(detail))) {
                free(pcm);
                h3_audio_latent_free(&latent);
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            free(pcm);
            if (latent.channels != 32 || latent.stereo != 2 ||
                latent.length < 1 ||
                (size_t)latent.length > SIZE_MAX / 2 / 32) {
                h3_audio_latent_free(&latent);
                h3_set_error(ctx,
                    "reference audio encoder produced invalid geometry");
                goto cleanup;
            }
            size_t elements = (size_t)latent.length * 2 * 32;
            if (condition_audio_elements > SIZE_MAX - elements ||
                condition_audio_elements + elements >
                    SIZE_MAX / sizeof(*condition_audio_rows)) {
                h3_audio_latent_free(&latent);
                h3_set_error(ctx, "reference audio row count overflows");
                goto cleanup;
            }
            float *grown = realloc(
                condition_audio_rows,
                (condition_audio_elements + elements) * sizeof(*grown));
            if (!grown) {
                h3_audio_latent_free(&latent);
                h3_set_error(ctx,
                    "out of memory packing reference audio conditions");
                goto cleanup;
            }
            condition_audio_rows = grown;
            float *rows = condition_audio_rows + condition_audio_elements;
            for (int stereo = 0; stereo < 2; stereo++)
                for (int time = 0; time < latent.length; time++)
                    for (int channel = 0; channel < 32; channel++) {
                        size_t source = ((size_t)channel * 2 +
                                         (size_t)stereo) * latent.length +
                                        (size_t)time;
                        size_t destination = ((size_t)stereo * latent.length +
                                              (size_t)time) * 32 +
                                             (size_t)channel;
                        rows[destination] = latent.values[source];
                    }
            condition_audio_elements += elements;
            total_audio_samples += (size_t)samples;
            layout_references[index].audio_t = latent.length;
            presentations[index].has_audio = 1;
            h3_audio_latent_free(&latent);
            if (progress.cancelled) goto cleanup;
        }
    } else {
        if (params->first_frame) {
            if (!h3_ffmpeg_read_image_f32(
                    params->first_frame, render_width, render_height,
                    H3_IMAGE_FIT_STRETCH, &condition_pixels[visual_count],
                    detail, sizeof(detail))) {
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            condition_widths[visual_count] = render_width;
            condition_heights[visual_count] = render_height;
            condition_frames[visual_count] = 1;
            visual_count++;
        }
        if (params->last_frame) {
            if (!h3_ffmpeg_read_image_f32(
                    params->last_frame, render_width, render_height,
                    H3_IMAGE_FIT_COVER, &condition_pixels[visual_count],
                    detail, sizeof(detail))) {
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            condition_widths[visual_count] = render_width;
            condition_heights[visual_count] = render_height;
            condition_frames[visual_count] = 1;
            visual_count++;
        }
        vision_output_count = visual_count;
    }
    if (vision_output_count) {
        vision_outputs = calloc(vision_output_count, sizeof(*vision_outputs));
        if (!vision_outputs) {
            h3_set_error(ctx, "out of memory allocating Qwen vision outputs");
            goto cleanup;
        }
    }

    if (visual_count || params->reference_count) {
        for (size_t image = 0; ref2va && image < visual_count; image++) {
            int image_latent_w, image_latent_h;
            h3_latent_canvas(condition_widths[image], condition_heights[image],
                             &image_latent_w, &image_latent_h);
            int image_latent_t = h3_video_encoder_latent_t(
                condition_frames[image]);
            size_t rows = (size_t)image_latent_t *
                          (size_t)image_latent_h * (size_t)image_latent_w / 4;
            if (rows > SIZE_MAX / 96 ||
                condition_video_elements > SIZE_MAX - rows * 96) {
                h3_set_error(ctx, "condition row count overflows");
                goto cleanup;
            }
            condition_video_elements += rows * 96;
        }
        if (condition_video_elements > SIZE_MAX /
                                       sizeof(*condition_video_rows)) {
            h3_set_error(ctx, "condition storage size overflows");
            goto cleanup;
        }
        condition_video_rows = condition_video_elements ?
            malloc(condition_video_elements * sizeof(*condition_video_rows)) : NULL;
        if (condition_video_elements && !condition_video_rows) {
            h3_set_error(ctx, "out of memory allocating visual condition rows");
            goto cleanup;
        }
        size_t condition_offset = 0;
        for (size_t image = 0; ref2va && image < visual_count; image++) {
            int image_latent_w, image_latent_h;
            h3_latent_canvas(condition_widths[image], condition_heights[image],
                             &image_latent_w, &image_latent_h);
            int image_latent_t = h3_video_encoder_latent_t(
                condition_frames[image]);
            size_t row_elements = (size_t)image_latent_t *
                                  (size_t)image_latent_h *
                                  (size_t)image_latent_w / 4 * 96;
            h3_video_latent latent;
            memset(&latent, 0, sizeof(latent));
            if (!h3_video_vae_encode(
                    vae_path, "h3_shaders.metal", condition_pixels[image],
                    condition_frames[image], condition_heights[image],
                    condition_widths[image],
                    h3_video_encoder_progress_bridge, &progress,
                    &latent, detail, sizeof(detail))) {
                h3_video_latent_free(&latent);
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            if (latent.time != image_latent_t ||
                latent.height != image_latent_h ||
                latent.width != image_latent_w) {
                h3_video_latent_free(&latent);
                h3_set_error(ctx,
                    "visual condition VAE produced unexpected latent geometry");
                goto cleanup;
            }
            float *rows = condition_video_rows + condition_offset;
            if (!h3_dit_patchify_video(
                    latent.values, 24, image_latent_t,
                    image_latent_h, image_latent_w,
                    rows, row_elements)) {
                h3_video_latent_free(&latent);
                h3_set_error(ctx, "cannot patchify visual condition latent");
                goto cleanup;
            }
            h3_video_latent_free(&latent);
            condition_offset += row_elements;
            if (progress.cancelled) goto cleanup;
        }
        if (condition_offset != condition_video_elements) {
            h3_set_error(ctx, "visual condition packing size mismatch");
            goto cleanup;
        }
        size_t vision_cursor = 0;
        for (size_t image = 0; image < visual_count; image++) {
            size_t reference_index = visual_reference_indices[image];
            if (!ref2va ||
                params->references[reference_index].kind == H3_REFERENCE_IMAGE) {
                if (!h3_vision_encode_bf16(
                        text_path, "h3_shaders.metal", condition_pixels[image],
                        1, condition_heights[image], condition_widths[image],
                        h3_vision_progress_bridge, &progress,
                        &vision_outputs[vision_cursor], detail, sizeof(detail))) {
                    h3_set_error(ctx, "%s", detail);
                    goto cleanup;
                }
                if (ref2va)
                    presentations[reference_index].vision =
                        &vision_outputs[vision_cursor];
                vision_cursor++;
            } else {
                size_t blocks = presentations[reference_index].vision_count;
                presentations[reference_index].vision =
                    &vision_outputs[vision_cursor];
                size_t samples = ((size_t)condition_frames[image] + 11) / 12;
                for (size_t block = 0; block < blocks; block++) {
                    size_t first_sample = 2 * block;
                    size_t second_sample = first_sample + 1 < samples ?
                                           first_sample + 1 : first_sample;
                    int first = (int)(first_sample * 12);
                    int second = (int)(second_sample * 12);
                    float *pair = h3_extract_vision_pair(
                        condition_pixels[image], condition_frames[image],
                        condition_heights[image], condition_widths[image],
                        first, second);
                    if (!pair) {
                        h3_set_error(ctx,
                            "out of memory extracting Qwen video pair");
                        goto cleanup;
                    }
                    int ok = h3_vision_encode_bf16(
                        text_path, "h3_shaders.metal", pair, 2,
                        condition_heights[image], condition_widths[image],
                        h3_vision_progress_bridge, &progress,
                        &vision_outputs[vision_cursor], detail, sizeof(detail));
                    free(pair);
                    if (!ok) {
                        h3_set_error(ctx, "%s", detail);
                        goto cleanup;
                    }
                    vision_cursor++;
                    if (progress.cancelled) goto cleanup;
                }
            }
            if (ref2va) { free(condition_pixels[image]); condition_pixels[image]=NULL; }
            if (progress.cancelled) goto cleanup;
        }
        if (vision_cursor != vision_output_count) {
            h3_set_error(ctx, "Qwen reference vision count mismatch");
            goto cleanup;
        }
        h3_progress_emit(&progress, "text encoder", 0, 50);
        int text_ok = ref2va ? h3_multimodal_encode_ref2va_bf16(
                tokenizer, text_path, "h3_shaders.metal", prompt,
                presentations, params->reference_count,
                h3_text_progress_bridge, &progress, &text,
                detail, sizeof(detail)) :
            h3_multimodal_encode_fl2va_bf16(
                tokenizer, text_path, "h3_shaders.metal", prompt,
                vision_outputs, visual_count,
                h3_text_progress_bridge, &progress, &text,
                detail, sizeof(detail));
        if (!text_ok) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
        for (size_t image = 0; image < vision_output_count; image++)
            h3_vision_output_free(&vision_outputs[image]);
    } else {
        if (!h3_tokenizer_encode(tokenizer, prompt, 1, &ids, &token_count,
                                 detail, sizeof(detail))) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
        h3_progress_emit(&progress, "text encoder", 0, 50);
        if (!h3_text_encode_bf16(
                text_path, "h3_shaders.metal", ids, token_count,
                h3_text_progress_bridge, &progress, &text,
                detail, sizeof(detail))) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
    }
    conditioned = visual_count != 0 || condition_audio_elements != 0;
    if (layout_guide_count) {
        layout_guides=calloc(layout_guide_count,sizeof(*layout_guides));
        if (!layout_guides) { h3_set_error(ctx,"out of memory preparing guides"); goto cleanup; }
        size_t gi=0;
        const char *endpoints[2]={params->first_frame,params->last_frame};
        for (int e=0;e<2;e++) if (endpoints[e]) {
            int frame=e ? temporal.frame_count-1 : 0;
            h3_guide g={H3_GUIDE_IMAGE,endpoints[e],NULL,frame,0,1,0};
            float *prepared=!ref2va ? condition_pixels[gi] : NULL;
            if(!ref2va) condition_pixels[gi]=NULL;
            if (!h3_encode_guide(&g,render_width,render_height,temporal.frame_count,
                vae_path,audio_vae_path,prepared,e ? H3_IMAGE_FIT_COVER : H3_IMAGE_FIT_STRETCH,
                &layout_guides[gi],&condition_video_rows,
                &condition_video_elements,&condition_audio_rows,&condition_audio_elements,detail,sizeof(detail))) {
                h3_set_error(ctx,"cannot encode endpoint guide: %s",detail); goto cleanup;
            }
            layout_guides[gi].source_index=e ? -2 : -1;
            gi++;
        }
        for (size_t i=0;i<params->guide_count;i++,gi++) {
            if (!h3_encode_guide(&params->guides[i],render_width,render_height,temporal.frame_count,
                vae_path,audio_vae_path,NULL,H3_IMAGE_FIT_COVER,&layout_guides[gi],&condition_video_rows,
                &condition_video_elements,&condition_audio_rows,&condition_audio_elements,detail,sizeof(detail))) {
                h3_set_error(ctx,"cannot encode guides[%zu]: %s",i,detail); goto cleanup;
            }
            layout_guides[gi].source_index=(int)i+1;
        }
        conditioned=1;
    }
    if (ctx->cache_enabled) {
        if (!h3_conditioning_cache_store(
                ctx, conditioning_key, &text,
                condition_video_rows, condition_video_elements,
                condition_audio_rows, condition_audio_elements,
                layout_references, ref2va ? params->reference_count : 0,
                conditioned))
            fprintf(stderr, "h3: warning: could not retain conditioning cache\n");
        else {
            if (layout_guide_count) {
                ctx->conditioning_guides=malloc(layout_guide_count*sizeof(*layout_guides));
                if (!ctx->conditioning_guides) h3_conditioning_cache_clear(ctx);
                else { memcpy(ctx->conditioning_guides,layout_guides,layout_guide_count*sizeof(*layout_guides));
                    ctx->conditioning_guide_count=layout_guide_count; }
            }
            fprintf(stderr, "h3: conditioning cache miss; stored exact BF16\n");
        }
    }
    }
    if (conditioned) h3_augment_conditions(params,layout_references,
        layout_guides,layout_guide_count,condition_video_rows);
    if (progress.cancelled) goto cleanup;

    h3_layout_spec spec = {(int)text.tokens, temporal.video_t, latent_h,
                           latent_w, temporal.audio_t, temporal.frame_count,
                           layout_guides, layout_guide_count,
                           layout_references,
                           ref2va ? params->reference_count : 0};
    if (!h3_layout_build(&spec, &layout, detail, sizeof(detail))) {
        h3_set_error(ctx, "%s", detail);
        goto cleanup;
    }
    float spatial_rope_scale = !params->use_reference_rope &&
        render_width == 256 && render_height == 256 ? 0.5f : 1.0f;
    if (ctx->cache_enabled && ctx->dit &&
        (!ctx->dit_key || strcmp(ctx->dit_key, prepared_key)) &&
        !h3_dit_can_prepare_request(ctx->dit, &layout))
        h3_prepared_cache_clear(ctx);
    if (ctx->cache_enabled && ctx->dit && ctx->dit_key &&
        !strcmp(ctx->dit_key, prepared_key)) {
        dit = ctx->dit;
        dit_is_cached = 1;
        if (!h3_dit_reset_run(
                dit, condition_video_rows, condition_video_elements,
                condition_audio_rows, condition_audio_elements,
                detail, sizeof(detail))) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
        fprintf(stderr, "h3: prepared DiT cache hit\n");
    } else if (ctx->cache_enabled && ctx->dit) {
        dit = ctx->dit;
        dit_is_cached = 1;
        free(ctx->dit_key); ctx->dit_key = NULL;
        if (!h3_dit_prepare_request(
                dit, &text, &layout, &sigmas, spatial_rope_scale,
                condition_video_rows, condition_video_elements,
                condition_audio_rows, condition_audio_elements,
                h3_dit_progress_bridge, &progress, detail, sizeof(detail))) {
            h3_set_error(ctx, "%s", detail);
            h3_prepared_cache_clear(ctx);
            dit = NULL;
            dit_is_cached = 0;
            goto cleanup;
        }
        fprintf(stderr, "h3: resident DiT weights reused; request re-prepared\n");
    } else if (conditioned) {
        dit = h3_dit_load_conditioned(
            dit_path, ctx->loras, "h3_shaders.metal", &text, &layout, &sigmas,
            (unsigned)params->dit_layers, (unsigned)params->core_reuse,
            params->token_reduction,
            params->ssd_streaming,
            spatial_rope_scale,
            params->use_slower_bf16_mlp,
            params->use_slower_bf16_qkv,
            params->use_slower_bf16_attention_output,
            params->use_slower_row_major_attention_output,
            params->use_slower_unfused_int8_inputs,
            params->use_slower_unfused_qkv_rope,
            params->use_slower_scalar_qkv_rms,
            params->use_slower_uncached_int8_scales,
            params->use_slower_dynamic_fc1_k,
            params->use_slower_grouped_quantizer,
            params->use_int8_row_fc2,
            condition_video_rows, condition_video_elements,
            condition_audio_rows, condition_audio_elements,
            h3_dit_progress_bridge, &progress, detail, sizeof(detail));
    } else {
        dit = h3_dit_load_t2va(
            dit_path, ctx->loras, "h3_shaders.metal", &text, &layout, &sigmas,
            (unsigned)params->dit_layers, (unsigned)params->core_reuse,
            params->token_reduction,
            params->ssd_streaming,
            spatial_rope_scale,
            params->use_slower_bf16_mlp,
            params->use_slower_bf16_qkv,
            params->use_slower_bf16_attention_output,
            params->use_slower_row_major_attention_output,
            params->use_slower_unfused_int8_inputs,
            params->use_slower_unfused_qkv_rope,
            params->use_slower_scalar_qkv_rms,
            params->use_slower_uncached_int8_scales,
            params->use_slower_dynamic_fc1_k,
            params->use_slower_grouped_quantizer,
            params->use_int8_row_fc2,
            h3_dit_progress_bridge, &progress, detail, sizeof(detail));
    }
    if (!dit) {
        h3_set_error(ctx, "%s", detail);
        goto cleanup;
    }
    if (ctx->cache_enabled && (!dit_is_cached || !ctx->dit_key)) {
        char *key_copy = strdup(prepared_key);
        char *model_copy = strdup(resident_key);
        if (!key_copy || !model_copy) {
            free(key_copy); free(model_copy);
            if (dit_is_cached) {
                ctx->dit = NULL;
                free(ctx->dit_model_key); ctx->dit_model_key = NULL;
                dit_is_cached = 0;
            }
            fprintf(stderr, "h3: warning: could not retain prepared DiT keys\n");
        } else {
            free(ctx->dit_key);
            free(ctx->dit_model_key);
            ctx->dit = dit;
            ctx->dit_key = key_copy;
            ctx->dit_model_key = model_copy;
            dit_is_cached = 1;
            fprintf(stderr, "h3: prepared DiT cache updated; model retained\n");
        }
    }
    h3_text_embedding_free(&text);
    free(condition_video_rows);
    condition_video_rows = NULL;
    free(condition_audio_rows);
    condition_audio_rows = NULL;
    if (progress.cancelled) goto cleanup;
    if (params->preview_denoise) {
        h3_progress_emit(&progress, "preview VAE load", 0, 36);
        preview_decoder = h3_acquire_video_decoder(
            ctx, decoder_key, vae_path, latent_h, latent_w,
            h3_preview_vae_progress_bridge, &progress,
            &decoder_is_cached, detail, sizeof(detail));
        if (preview_decoder && ctx->video_decoder == preview_decoder)
            h3_progress_emit(&progress, "preview VAE load", 36, 36);
        if (!preview_decoder) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
        live_preview.progress = &progress;
        live_preview.decoder = preview_decoder;
        live_preview.latent_t = temporal.video_t;
        live_preview.latent_h = latent_h;
        live_preview.latent_w = latent_w;
        live_preview.output_frames = temporal.frame_count;
        live_preview.output_width = params->width;
        live_preview.output_height = params->height;
        if (progress.cancelled) goto cleanup;
    }
    size_t video_count = h3_dit_video_elements(dit);
    size_t audio_count = h3_dit_audio_elements(dit);
    video = malloc(video_count * sizeof(*video));
    audio = malloc(audio_count * sizeof(*audio));
    if (!video || !audio) {
        h3_set_error(ctx, "out of memory allocating joint H3 noise");
        goto cleanup;
    }
    /* The released server initializes each modality from a separate generator
     * carrying the same requested seed. */
    h3_rng video_rng, audio_rng;
    h3_rng_seed(&video_rng, params->seed);
    h3_rng_seed(&audio_rng, params->seed);
    h3_rng_fill_normal(&video_rng, video, video_count);
    h3_rng_fill_normal(&audio_rng, audio, audio_count);
    int denoised=0;
    if (controlled) {
        unsigned char current_digest[32];
        if (video_count!=config.video_count || audio_count!=config.audio_count ||
            !h3_exact_digest(ctx,prompt,params,render_width,render_height,ref2va,
                             dependency_paths,current_digest) ||
            ctx->file_digest_revision!=dependency_revision ||
            memcmp(current_digest,config.request_digest,sizeof(current_digest))) {
            h3_set_error(ctx,"dependency_changed: exact checkpoint dependencies changed during preparation");goto cleanup;
        }
        if (params->initial_latent_path) {
            float sv=initial_video_sigma, sa=initial_audio_sigma;
            for (size_t i=0;i<video_count;i++) video[i]=(1-sv)*initial.video[i]+sv*video[i];
            for (size_t i=0;i<audio_count;i++) audio[i]=(1-sa)*initial.audio[i]+sa*audio[i];
        }
        if (!sampler_state)
            sampler_state=h3_sampler_state_start(&config,video,audio,initial.video,masks.video,initial.audio,masks.audio,detail,sizeof(detail));
        if (!sampler_state) { h3_set_error(ctx,"checkpoint_incompatible: %s",detail); goto cleanup; }
        h3_transition_control control={params,sigmas.steps,0};
        denoised=h3_dit_denoise_state(dit,sampler_state,sigmas.steps,h3_dit_progress_bridge,&progress,
            preview_decoder ? h3_deliver_denoise_preview : NULL,preview_decoder ? &live_preview : NULL,
            h3_transition_observe,&control,detail,sizeof(detail));
        int action=control.action;
        if(action) denoised=1; /* Observer stopped only after a committed transition. */
        if (denoised && (!h3_exact_digest(ctx,prompt,params,render_width,render_height,ref2va,
                                         dependency_paths,current_digest) ||
                         ctx->file_digest_revision!=dependency_revision ||
                         memcmp(current_digest,config.request_digest,sizeof(current_digest)))) {
            h3_set_error(ctx,"dependency_changed: exact checkpoint dependencies changed during sampling");goto cleanup;
        }
        if (denoised && params->checkpoint_path &&
            !h3_sampler_state_save(sampler_state,params->checkpoint_path,detail,sizeof(detail))) {
            h3_set_error(ctx,"output_failed: checkpoint: %s",detail); goto cleanup;
        }
        if (denoised && action) {
            if (action==1 && !params->checkpoint_path) {
                h3_set_error(ctx,"output_failed: pause requires checkpoint_path"); goto cleanup;
            }
            result=calloc(1,sizeof(*result));
            if (!result) { h3_set_error(ctx,"out of memory creating stopped result"); goto cleanup; }
            result->status=action==1 ? 1 : 2; result->width=params->width; result->height=params->height;
            result->frames=temporal.frame_count; result->fps=H3_FPS; result->seed=params->seed;
            goto cleanup;
        }
        if (denoised) {
            memcpy(video,h3_sampler_state_video(sampler_state),video_count*sizeof(float));
            memcpy(audio,h3_sampler_state_audio(sampler_state),audio_count*sizeof(float));
        }
    } else denoised = params->scheduler != H3_SCHEDULER_SERVING ?
        h3_dit_denoise_simple_preview(
            dit, params->sampler, params->seed, params->video_shift, params->audio_shift,
            video, audio, h3_dit_progress_bridge, &progress,
            preview_decoder ? h3_deliver_denoise_preview : NULL,
            preview_decoder ? &live_preview : NULL, detail, sizeof(detail)) :
        h3_dit_denoise_euler_preview(
            dit, video, audio, params->denoise_reuse,
            h3_dit_progress_bridge, &progress,
            preview_decoder ? h3_deliver_denoise_preview : NULL,
            preview_decoder ? &live_preview : NULL, detail, sizeof(detail));
    if (!denoised) {
        if (!live_preview.failed) h3_set_error(ctx, "%s", detail);
        if (dit_is_cached) {
            ctx->dit = NULL;
            free(ctx->dit_key);
            ctx->dit_key = NULL;
            free(ctx->dit_model_key);
            ctx->dit_model_key = NULL;
            dit_is_cached = 0;
        }
        goto cleanup;
    }
    if (!dit_is_cached) h3_dit_free(dit);
    dit = NULL;
    if (progress.cancelled) goto cleanup;
    if (params->latent_output_path) {
        h3_latent clean={.video_frames=(uint64_t)temporal.video_t,.height=(uint64_t)latent_h,
            .width=(uint64_t)latent_w,.audio_frames=(uint64_t)temporal.audio_t,
            .video_count=video_count,.audio_count=audio_count,.video=video,.audio=audio,
            .source_frames=(uint64_t)temporal.frame_count,.temporal_mapping=H3_LATENT_TEMPORAL_NATIVE};
        if (!h3_latent_save(params->latent_output_path,&clean,detail,sizeof(detail))) {
            h3_set_error(ctx,"output_failed: latent: %s",detail); goto cleanup;
        }
    }
    h3_progress_emit(&progress, "audio VAE", 0, 7);
    if (!h3_audio_vae_decode(audio_vae_path, "h3_shaders.metal", audio,
                             temporal.audio_t, h3_audio_vae_progress_bridge,
                             &progress, &waveform, detail, sizeof(detail))) {
        h3_set_error(ctx, "%s", detail);
        goto cleanup;
    }
    free(audio);
    audio = NULL;
    if (progress.cancelled) goto cleanup;
    if (!preview_decoder && ctx->cache_enabled) {
        h3_progress_emit(&progress, "video VAE load", 0, 36);
        preview_decoder = h3_acquire_video_decoder(
            ctx, decoder_key, vae_path, latent_h, latent_w,
            h3_vae_progress_bridge, &progress,
            &decoder_is_cached, detail, sizeof(detail));
        if (preview_decoder && ctx->video_decoder == preview_decoder)
            h3_progress_emit(&progress, "video VAE load", 36, 36);
        if (!preview_decoder) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
    }
    int video_ok = preview_decoder ?
        h3_video_vae_decoder_decode(
            preview_decoder, video, temporal.video_t, &frames,
            detail, sizeof(detail)) :
        h3_video_vae_decode(
            vae_path, "h3_shaders.metal", video,
            temporal.video_t, latent_h, latent_w,
            h3_vae_progress_bridge, &progress, &frames,
            detail, sizeof(detail));
    if (!video_ok) {
        h3_set_error(ctx, "%s", detail);
        goto cleanup;
    }
    free(video);
    video = NULL;
    if (progress.cancelled) goto cleanup;
    size_t rgb_count = (size_t)frames.frames * (size_t)frames.height *
                       (size_t)frames.width * 3;
    rgb8 = h3_rgb_f32_to_u8(frames.rgb, rgb_count);
    if (!rgb8) {
        h3_set_error(ctx, "out of memory converting generated RGB frames");
        goto cleanup;
    }
    int output_width = frames.width;
    int output_height = frames.height;
    if (output_width != params->width || output_height != params->height) {
        uint8_t *resized = NULL;
        if (!h3_resize_rgb24_high_quality(
                rgb8, frames.frames, output_width, output_height,
                params->width, params->height, &resized)) {
            h3_set_error(ctx, "cannot resize generated RGB frames");
            goto cleanup;
        }
        free(rgb8);
        rgb8 = resized;
        output_width = params->width;
        output_height = params->height;
    }
    if (params->on_frame) {
        size_t frame_bytes = (size_t)output_width * (size_t)output_height * 3;
        for (int index = 0; index < frames.frames; index++) {
            h3_frame frame = {output_width, output_height, output_width * 3,
                              rgb8 + (size_t)index * frame_bytes,
                              index, frames.frames, -1, 0};
            if (params->on_frame(&frame, params->callback_opaque)) {
                h3_set_error(ctx, "generation cancelled while delivering frame %d",
                             index);
                goto cleanup;
            }
        }
    }
    if (params->output_path && *params->output_path) {
        h3_progress_emit(&progress, "FFmpeg", 0, frames.frames);
        if (!h3_ffmpeg_write_av_rgb24_f32(
                params->output_path, rgb8, frames.frames, output_width,
                output_height, H3_FPS, waveform.pcm, waveform.samples,
                waveform.channels, waveform.sample_rate,
                detail, sizeof(detail))) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
        h3_progress_emit(&progress, "FFmpeg", frames.frames, frames.frames);
    }
    result = calloc(1, sizeof(*result));
    if (!result) {
        h3_set_error(ctx, "out of memory creating generation result");
        goto cleanup;
    }
    result->width = output_width;
    result->height = output_height;
    result->frames = frames.frames;
    result->fps = H3_FPS;
    result->sample_rate = waveform.sample_rate;
    result->seed = params->seed;

cleanup:
    free(conditioning_key);
    free(prepared_key);
    free(resident_key);
    free(decoder_key);
    free(model_identity);
    free(tokenizer_path); free(text_path); free(dit_path); free(vae_path);
    free(audio_vae_path);
    free(vae_config_path);
    h3_tokenizer_free(tokenizer);
    h3_tokenizer_ids_free(ids);
    for (size_t image = 0; image < visual_capacity; image++) {
        if (condition_pixels) free(condition_pixels[image]);
    }
    for (size_t image = 0; image < vision_output_count; image++)
        if (vision_outputs) h3_vision_output_free(&vision_outputs[image]);
    for (size_t index = 0; index < params->reference_count; index++)
        if (presentation_timestamps) free(presentation_timestamps[index]);
    free(condition_pixels);
    free(condition_widths);
    free(condition_heights);
    free(condition_frames);
    free(visual_reference_indices);
    free(reference_visual_indices);
    free(vision_outputs);
    free(presentations);
    free(presentation_timestamps);
    free(layout_references);
    free(layout_guides);
    h3_sampler_state_free(sampler_state);
    h3_latent_free(&initial); h3_latent_free(&masks);
    free(condition_video_rows);
    free(condition_audio_rows);
    h3_text_embedding_free(&text);
    h3_layout_free(&layout);
    if (!dit_is_cached) h3_dit_free(dit);
    if (!decoder_is_cached) h3_video_vae_decoder_free(preview_decoder);
    free(video); free(audio); free(rgb8);
    h3_video_frames_free(&frames);
    h3_audio_waveform_free(&waveform);
    return result;
}

h3_result *h3_resume(h3_ctx *ctx,const char *prompt,const h3_params *params) {
    if (!params || !params->resume_path) { h3_set_error(ctx,"resume requires resume_path"); return NULL; }
    return h3_generate(ctx,prompt,params);
}

h3_result *h3_encode_media(h3_ctx *ctx,const char *input,int streams,const h3_params *p) {
    if (!ctx) return NULL;
    ctx->error[0]=0;
    if (!input || !p || streams<1 || streams>3 || !p->latent_output_path) {
        h3_set_error(ctx,"invalid_request: encode requires input, stream policy and latent_output_path"); return NULL;
    }
    if(!h3_valid_params(ctx,p)) {
        if(!strcmp(ctx->error_code,"engine_failed"))snprintf(ctx->error_code,sizeof(ctx->error_code),"invalid_request");
        return NULL;
    }
    int ref=p->reference_count!=0 || ctx->lora_target==H3_LORA_REF2VA;
    int width=p->render_width ? p->render_width : p->width;
    int height=p->render_height ? p->render_height : p->height;
    char *vp=h3_path(ctx->model_dir,ref ? "Ref2VA/video_vae/source" : "FL2VA/video_vae/source");
    char *ap=h3_path(ctx->model_dir,ref ? "Ref2VA/audio_vae" : "FL2VA/audio_vae");
    h3_latent latent={0}; h3_video_latent vz={0}; h3_audio_latent az={0};
    float *pixels=NULL,*pcm=NULL; int frames=0,samples=0; char error[512]={0}; h3_result *r=NULL;
    if (!vp || !ap) { h3_set_error(ctx,"out of memory resolving encoder paths"); goto done; }
    if (streams&1) {
        if (!h3_ffmpeg_read_video_f32(input,width,height,p->frames,&pixels,&frames,error,sizeof(error)) ||
            frames!=p->frames || !h3_video_vae_encode(vp,"h3_shaders.metal",pixels,frames,height,width,
                NULL,NULL,&vz,error,sizeof(error))) { h3_set_error(ctx,"encode video: %s",error); goto done; }
        int target=h3_video_latent_t(frames); size_t area=(size_t)vz.height*vz.width;
        latent.video_count=(size_t)24*target*area;
        latent.video=malloc(latent.video_count*sizeof(float));
        if (!latent.video) { h3_set_error(ctx,"out of memory mapping video temporal convention"); goto done; }
        /* Explicit endpoint-aligned linear temporal conversion; never reinterpret encoder T. */
        for(int c=0;c<24;c++) for(int t=0;t<target;t++) {
            double at=target>1 ? (double)t*(vz.time-1)/(target-1) : 0;
            int lo=(int)floor(at),hi=lo+1<vz.time ? lo+1 : lo; float w=(float)(at-lo);
            for(size_t i=0;i<area;i++) latent.video[((size_t)c*target+t)*area+i]=
                (1-w)*vz.values[((size_t)c*vz.time+lo)*area+i]+w*vz.values[((size_t)c*vz.time+hi)*area+i];
        }
        latent.video_frames=target; latent.height=vz.height; latent.width=vz.width;
        latent.source_frames=frames; latent.temporal_mapping=1;
    }
    if (streams&2) {
        int limit = streams == 2 ? 32000 * 16 :
            (int)llround(p->frames * 32000.0 / 24.0);
        if (!h3_ffmpeg_read_audio_f32(input,limit,streams==3,&pcm,&samples,error,sizeof(error)) ||
            !h3_audio_vae_encode(ap,"h3_shaders.metal",pcm,samples,NULL,NULL,&az,error,sizeof(error))) {
            h3_set_error(ctx,"encode audio: %s",error); goto done;
        }
        if(streams==3) {
            int target=h3_temporal(p->frames).audio_t;
            if(az.length<target || az.length>target+1) {
                h3_set_error(ctx,"latent_incompatible: audiovisual source audio does not span the target duration");goto done;
            }
            /* The encoder right-pads; drop only the final excess padded latent tick. */
            if(az.length!=target) for(int i=0;i<64;i++) memmove(az.values+(size_t)i*target,
                az.values+(size_t)i*az.length,(size_t)target*sizeof(float));
            az.length=target;
        }
        latent.audio=az.values; az.values=NULL; latent.audio_frames=az.length;
        latent.audio_count=(size_t)64*az.length;
    }
    if (!h3_latent_save(p->latent_output_path,&latent,error,sizeof(error))) {
        h3_set_error(ctx,"output_failed: %s",error); goto done;
    }
    r=calloc(1,sizeof(*r));
    if (!r) { h3_set_error(ctx,"out of memory creating encode result"); goto done; }
    r->width=(streams&1) ? width : 0; r->height=(streams&1) ? height : 0;
    r->frames=frames; r->fps=24; r->sample_rate=(streams&2) ? 32000 : 0;
done:
    free(vp);free(ap);free(pixels);free(pcm);
    h3_video_latent_free(&vz);h3_audio_latent_free(&az);h3_latent_free(&latent);return r;
}

static void h3_le16(unsigned char *p,uint16_t x) { p[0]=(unsigned char)x;p[1]=(unsigned char)(x>>8); }
static void h3_le32(unsigned char *p,uint32_t x) { for(int i=0;i<4;i++)p[i]=(unsigned char)(x>>(8*i)); }
static int h3_write_wave(const char *path,const h3_audio_waveform *a,char *error,size_t n) {
    if (a->channels!=2 || a->samples<1 || (uint64_t)a->samples*8>UINT32_MAX-36) {
        snprintf(error,n,"invalid stereo waveform");return 0;
    }
    size_t len=strlen(path)+16;char *temp=malloc(len);
    if(!temp) {snprintf(error,n,"out of memory creating audio output");return 0;}
    snprintf(temp,len,"%s.tmp.XXXXXX",path);int fd=mkstemp(temp);
    if(fd<0) {snprintf(error,n,"cannot create audio output: %s",strerror(errno));free(temp);return 0;}
    FILE *f=fdopen(fd,"wb");if(!f){close(fd);unlink(temp);free(temp);return 0;}
    unsigned char h[44]={0};memcpy(h,"RIFF",4);h3_le32(h+4,36+(uint32_t)a->samples*8);
    memcpy(h+8,"WAVEfmt ",8);h3_le32(h+16,16);h3_le16(h+20,3);h3_le16(h+22,2);
    h3_le32(h+24,(uint32_t)a->sample_rate);h3_le32(h+28,(uint32_t)a->sample_rate*8);
    h3_le16(h+32,8);h3_le16(h+34,32);memcpy(h+36,"data",4);h3_le32(h+40,(uint32_t)a->samples*8);
    int ok=fwrite(h,1,sizeof(h),f)==sizeof(h);
    unsigned char chunk[4096];
    for(int start=0;ok && start<a->samples;start+=512) {
        int count=a->samples-start<512 ? a->samples-start : 512;
        for(int i=0;i<count;i++) for(int c=0;c<2;c++) {
            uint32_t bits;memcpy(&bits,a->pcm+(size_t)c*a->samples+start+i,4);
            h3_le32(chunk+((size_t)i*2+c)*4,bits);
        }
        ok=fwrite(chunk,8,(size_t)count,f)==(size_t)count;
    }
    if(ok)ok=fflush(f)==0 && fsync(fd)==0;
    if(fclose(f)!=0)ok=0;
    if(ok)ok=rename(temp,path)==0;
    if(!ok){snprintf(error,n,"cannot publish audio output: %s",strerror(errno));unlink(temp);}
    free(temp);return ok;
}

static h3_result *h3_decode_latent_output(h3_ctx *ctx,const char *input,int diagnostic,
    const h3_params *p,int target_width,int target_height) {
    if (!ctx) return NULL;
    ctx->error[0]=0;
    if (!input || !p) { h3_set_error(ctx,"invalid_request: decode requires input and params"); return NULL; }
    h3_latent latent={0};char error[512]={0}; h3_video_frames frames={0};h3_audio_waveform audio={0};
    uint8_t *rgb=NULL;h3_result *r=NULL;
    h3_sampler_state *preview=NULL;
    int output_width=0,output_height=0;
    int ref=p->reference_count!=0 || ctx->lora_target==H3_LORA_REF2VA;
    char *vp=h3_path(ctx->model_dir,ref ? "Ref2VA/video_vae/source" : "FL2VA/video_vae/source");
    char *ap=h3_path(ctx->model_dir,ref ? "Ref2VA/audio_vae" : "FL2VA/audio_vae");
    if (!vp || !ap) { h3_set_error(ctx,"out of memory resolving decoder paths");goto done; }
    if (!h3_latent_load(input,&latent,error,sizeof(error))) {
        if(!diagnostic || !h3_valid_params(ctx,p)) {
            if(!ctx->error[0]) h3_set_error(ctx,"latent_incompatible: %s",error);goto done;
        }
        preview=h3_sampler_state_load_preview(input,error,sizeof(error));
        if(!preview) {h3_set_error(ctx,"latent_incompatible: %s",error);goto done;}
        const h3_sampler_config *config=h3_sampler_state_config(preview);
        h3_temporal_shape temporal=h3_temporal(p->frames);
        latent.height=(uint64_t)(p->render_height ? p->render_height : p->height)/16;
        latent.width=(uint64_t)(p->render_width ? p->render_width : p->width)/16;
        latent.video_frames=temporal.video_t;latent.audio_frames=temporal.audio_t;
        latent.video_count=(size_t)24*latent.video_frames*latent.height*latent.width;
        latent.audio_count=(size_t)64*latent.audio_frames;
        if(config->video_count!=latent.video_count || config->audio_count!=latent.audio_count) {
            h3_set_error(ctx,"latent_incompatible: diagnostic checkpoint target shape mismatch");goto done;
        }
        latent.video=(float *)h3_sampler_state_video(preview);
        latent.audio=(float *)h3_sampler_state_audio(preview);
    }
    if (latent.audio_count && !h3_audio_vae_decode(ap,"h3_shaders.metal",latent.audio,
        (int)latent.audio_frames,NULL,NULL,&audio,error,sizeof(error))) {
        h3_set_error(ctx,"decode audio: %s",error);goto done;
    }
    if (latent.video_count) {
        if (!h3_video_vae_decode(vp,"h3_shaders.metal",latent.video,(int)latent.video_frames,
            (int)latent.height,(int)latent.width,NULL,NULL,&frames,error,sizeof(error))) {
            h3_set_error(ctx,"decode video: %s",error);goto done;
        }
        rgb=h3_rgb_f32_to_u8(frames.rgb,(size_t)frames.frames*frames.height*frames.width*3);
        if (!rgb) { h3_set_error(ctx,"out of memory converting decoded frames"); goto done; }
        output_width=frames.width;output_height=frames.height;
        if(target_width && (output_width!=target_width || output_height!=target_height)) {
            uint8_t *resized=NULL;
            if(!h3_resize_rgb24_high_quality(rgb,frames.frames,output_width,output_height,
                target_width,target_height,&resized)) {
                h3_set_error(ctx,"cannot resize terminal generation frames");goto done;
            }
            free(rgb);rgb=resized;output_width=target_width;output_height=target_height;
        }
        if (p->on_frame) for(int i=0;i<frames.frames;i++) {
            h3_frame f={output_width,output_height,output_width*3,
                rgb+(size_t)i*output_width*output_height*3,i,frames.frames,-1,0};
            if (p->on_frame(&f,p->callback_opaque)) { h3_set_error(ctx,"cancelled: decode frame delivery"); goto done; }
        }
    }
    if (p->output_path) {
        int ok=!latent.video_count ? h3_write_wave(p->output_path,&audio,error,sizeof(error)) :
            latent.audio_count ? h3_ffmpeg_write_av_rgb24_f32(p->output_path,rgb,frames.frames,
            output_width,output_height,24,audio.pcm,audio.samples,audio.channels,audio.sample_rate,error,sizeof(error)) :
            h3_ffmpeg_write_rgb24(p->output_path,rgb,frames.frames,output_width,output_height,24,error,sizeof(error));
        if (!ok) { h3_set_error(ctx,"output_failed: %s",error);goto done; }
    }
    r=calloc(1,sizeof(*r));
    if (!r) { h3_set_error(ctx,"out of memory creating decode result");goto done; }
    r->width=output_width;r->height=output_height;r->frames=frames.frames;r->fps=24;
    r->sample_rate=audio.sample_rate;r->seed=p->seed;
    r->diagnostic_preview=preview!=NULL;
done:
    if(preview) {latent.video=NULL;latent.audio=NULL;h3_sampler_state_free(preview);}
    free(vp);free(ap);free(rgb);h3_latent_free(&latent);
    h3_video_frames_free(&frames);h3_audio_waveform_free(&audio);return r;
}

h3_result *h3_decode_latent(h3_ctx *ctx,const char *input,int diagnostic,const h3_params *params) {
    /* Standalone decoding preserves the bundle's native canvas. */
    return h3_decode_latent_output(ctx,input,diagnostic,params,0,0);
}

void h3_result_free(h3_result *result) {
    free(result);
}
