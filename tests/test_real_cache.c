/* Opt-in real-weight regression: see usage below. Never part of make test. */
#include "h3.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define HASH_INIT UINT64_C(14695981039346656037)
#define CASE_COUNT 11

typedef struct {
    uint64_t rgb, audio, audio_bytes;
    int frames, fps, sample_rate, loads, invalid, cancel, cancelled;
    int width, height, expected_frames;
    uint64_t seed;
} capture;

typedef struct {
    const char *name, *prompt;
    int steps, width, height, frames;
    uint64_t seed;
    capture warm;
} scenario;

static uint64_t hash_bytes(uint64_t hash, const unsigned char *bytes, size_t n) {
    for (size_t i = 0; i < n; i++) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int on_frame(const h3_frame *frame, void *opaque) {
    capture *c = opaque;
    if (frame->denoise_step >= 0) return 0;
    if (!frame->rgb || frame->width != c->width || frame->height != c->height ||
        frame->stride < c->width * 3 || frame->frame_index != c->frames ||
        frame->frame_count != c->expected_frames) {
        c->invalid = 1;
        return 1;
    }
    for (int y = 0; y < frame->height; y++)
        c->rgb = hash_bytes(c->rgb, frame->rgb + (size_t)y * (size_t)frame->stride,
                            (size_t)frame->width * 3);
    c->frames++;
    return 0;
}

static int on_progress(const char *phase, int completed, int total, void *opaque) {
    capture *c = opaque;
    if (!strcmp(phase, "load transformer core") && total > 0 && completed == total)
        c->loads++;
    if (c->cancel && !strcmp(phase, "denoise") && completed > 0) {
        c->cancelled = 1;
        return 1;
    }
    return 0;
}

/* Decode ALL samples, without a shell, including explicit stream selection.
 * PCM hash is after MP4 audio encoding/decoding, not pre-encoding native PCM. */
static int audio_hash(const char *path, capture *c) {
    int fds[2];
    if (pipe(fds)) return 0;
    fflush(NULL);
    pid_t child = fork();
    if (child < 0) {
        close(fds[0]); close(fds[1]);
        return 0;
    }
    if (child == 0) {
        close(fds[0]);
        if (dup2(fds[1], STDOUT_FILENO) < 0) _exit(126);
        close(fds[1]);
        execlp("ffmpeg", "ffmpeg", "-nostdin", "-v", "error", "-i", path,
               "-map", "0:a:0", "-vn", "-c:a", "pcm_f32le", "-f", "f32le",
               "pipe:1", (char *)NULL);
        _exit(127);
    }
    close(fds[1]);
    unsigned char buffer[16384];
    c->audio = HASH_INIT;
    int ok = 1;
    for (;;) {
        ssize_t n = read(fds[0], buffer, sizeof(buffer));
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) { ok = 0; break; }
        if (!n) break;
        c->audio = hash_bytes(c->audio, buffer, (size_t)n);
        c->audio_bytes += (uint64_t)n;
    }
    close(fds[0]);
    int status = 0;
    pid_t waited;
    do { waited = waitpid(child, &status, 0); } while (waited < 0 && errno == EINTR);
    return ok && waited == child && WIFEXITED(status) && !WEXITSTATUS(status) &&
           c->audio_bytes > 0;
}

static h3_ctx *load_context(const char *root, const char *lora, int ref2va) {
    h3_lora_spec spec = {lora, 1.0f};
    int active = strcmp(lora, "none") != 0;
    h3_ctx *ctx = h3_load_dir(root, active ? &spec : NULL, active ? 1 : 0,
                            ref2va ? H3_LORA_REF2VA : H3_LORA_FL2VA);
    if (ctx) h3_cache_set_enabled(ctx, 1);
    else fprintf(stderr, "FAIL: h3_load_dir\n");
    return ctx;
}

static int generate(h3_ctx *ctx, const scenario *s, const char *dir,
                    const char *prefix, const h3_reference *refs, size_t nrefs,
                    int cancel, capture *c, FILE *manifest) {
    char path[4096];
    int length = snprintf(path, sizeof(path), "%s/%s-%s.mp4", dir, prefix, s->name);
    if (length < 0 || (size_t)length >= sizeof(path)) return 0;
    *c = (capture){.rgb = HASH_INIT, .cancel = cancel, .width = s->width,
                   .height = s->height, .expected_frames = s->frames};
    h3_params p = H3_PARAMS_DEFAULT;
    p.width = s->width; p.height = s->height; p.frames = s->frames;
    p.steps = s->steps; p.seed = s->seed;
    p.sampler = H3_SAMPLER_RES_MULTISTEP; p.scheduler = H3_SCHEDULER_SIMPLE;
    p.output_path = path; p.references = refs; p.reference_count = nrefs;
    p.on_frame = on_frame; p.on_progress = on_progress; p.callback_opaque = c;
    h3_result *result = h3_generate(ctx, s->prompt, &p);
    if (cancel) {
        int ok = !result && c->cancelled && !c->frames;
        if (!ok) fprintf(stderr, "FAIL: cancellation did not stop generation\n");
        h3_result_free(result);
        return ok;
    }
    if (!result) {
        fprintf(stderr, "FAIL: %s/%s: %s\n", prefix, s->name, h3_last_error(ctx));
        return 0;
    }
    c->fps = result->fps; c->sample_rate = result->sample_rate; c->seed = result->seed;
    int ok = !c->invalid && c->frames == s->frames && result->frames == s->frames &&
        result->width == s->width && result->height == s->height &&
        result->fps == 24 && result->sample_rate > 0 && result->seed == s->seed;
    h3_result_free(result);
    if (!ok || !audio_hash(path, c)) {
        fprintf(stderr, "FAIL: invalid frames/metadata or decoded audio: %s\n", path);
        return 0;
    }
    fprintf(manifest, "%s\t%s\t%d\t%d\t%d\t%d\t%d\t%" PRIu64
            "\t%016" PRIx64 "\t%016" PRIx64 "\t%" PRIu64 "\t%d\t%s\n",
            prefix, s->name, s->width, s->height, c->frames, c->fps, c->sample_rate,
            c->seed, c->rgb, c->audio, c->audio_bytes, c->loads, path);
    fflush(manifest);
    return 1;
}

static int same(const capture *a, const capture *b) {
    return a->rgb == b->rgb && a->audio == b->audio &&
        a->audio_bytes == b->audio_bytes && a->frames == b->frames &&
        a->fps == b->fps && a->sample_rate == b->sample_rate && a->seed == b->seed;
}

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5 && argc != 6) {
        fprintf(stderr, "usage: %s MODEL_ROOT LORA_PATH|none OUTPUT_DIR [REF_IMAGE [REF_AUDIO]]\n",
                argv[0]);
        return 2;
    }
    if (mkdir(argv[3], 0755) && errno != EEXIST) { perror("mkdir"); return 2; }
    /* Resolve output directory so a leading '-' cannot become an ffmpeg option. */
    char *dir = realpath(argv[3], NULL);
    if (!dir) { perror("realpath"); return 2; }
    char manifest_path[4096];
    int length = snprintf(manifest_path, sizeof(manifest_path), "%s/results.tsv", dir);
    if (length < 0 || (size_t)length >= sizeof(manifest_path)) { free(dir); return 2; }
    FILE *manifest = fopen(manifest_path, "w");
    if (!manifest) { perror("manifest"); free(dir); return 2; }
    fprintf(manifest, "context\tcase\twidth\theight\tframes\tfps\tsample_rate\tseed"
            "\trgb_fnv1a64\tdecoded_audio_fnv1a64\taudio_bytes\tcore_loads\tfile\n");
    h3_reference refs[2] = {{H3_REFERENCE_IMAGE, argc >= 5 ? argv[4] : NULL, NULL, 0},
                            {H3_REFERENCE_AUDIO, argc >= 6 ? argv[5] : NULL, NULL, 0}};
    size_t nrefs = argc >= 5 ? (size_t)(argc - 4) : 0;
    const char *short_prompt = "A red sailboat moves across a calm blue lake.";
    const char *long_prompt = "A small golden robot plays a grand piano in a spacious"
        " old concert hall. The camera slowly circles the piano, showing polished"
        " wooden floors, velvet seats, and warm sunlight through tall windows."
        " Its articulated hands press the keys as clear piano music echoes softly.";
    scenario cases[CASE_COUNT] = {
        {"cold", short_prompt, 5, 512, 512, 22, 42, {0}},
        {"seed", short_prompt, 5, 512, 512, 22, 314159, {0}},
        {"prompt", long_prompt, 5, 512, 512, 22, 42, {0}},
        {"steps", long_prompt, 7, 512, 512, 22, 42, {0}},
        {"steps-back", long_prompt, 5, 512, 512, 22, 42, {0}},
        {"shape", long_prompt, 5, 576, 576, 39, 42, {0}},
        {"shape-back", long_prompt, 5, 512, 512, 22, 42, {0}},
        {"clear", long_prompt, 5, 512, 512, 22, 42, {0}},
        {"recover", long_prompt, 5, 512, 512, 22, 42, {0}},
        {"reference-audio-off", long_prompt, 5, 512, 512, 22, 42, {0}},
        {"reference-audio-back", long_prompt, 5, 512, 512, 22, 42, {0}}
    };
    int case_count = nrefs > 1 ? CASE_COUNT : CASE_COUNT - 2;
    int ok = 1;
    h3_ctx *ctx = load_context(argv[1], argv[2], nrefs != 0);
    if (!ctx) ok = 0;
    for (int i = 0; ok && i < case_count; i++) {
        if (i == 7) {
            h3_cache_clear(ctx);
            h3_cache_info info = {0};
            h3_cache_get_info(ctx, &info);
            if (info.prepared_dit || info.embedding_entries || info.video_decoder) {
                fprintf(stderr, "FAIL: cache clear retained public cache entries\n");
                ok = 0; break;
            }
        }
        if (i == 8) {
            capture cancelled;
            ok = generate(ctx, &cases[i], dir, "cancel", refs, nrefs, 1,
                          &cancelled, manifest);
            if (!ok) break;
        }
        size_t count = i == CASE_COUNT - 2 ? 1 : nrefs;
        ok = generate(ctx, &cases[i], dir, "warm", refs, count, 0,
                      &cases[i].warm, manifest);
        /* Exact full-depth native path; no gate-pruning or precision changes.
         * Recovery may legitimately reload after cancellation invalidates state. */
        if (ok && i != 8 && cases[i].warm.loads != (i == 0 || i == 7 ? 1 : 0)) {
            fprintf(stderr, "FAIL: unexpected resident core reload in %s: %d\n",
                    cases[i].name, cases[i].warm.loads);
            ok = 0;
        }
    }
    h3_free(ctx);
    /* Never keep two complete model contexts resident simultaneously. Collect
     * every warm result first, then compare with isolated fresh contexts. */
    for (int i = 0; ok && i < case_count; i++) {
        /* Restored settings reuse their already-verified fresh oracle, not a
         * second identical render. The warm transition itself still ran. */
        int oracle = -1;
        size_t ref_count = i == CASE_COUNT - 2 ? 1 : nrefs;
        for (int j = 0; j < i; j++) {
            size_t prior_refs = j == CASE_COUNT - 2 ? 1 : nrefs;
            if (ref_count == prior_refs && cases[i].steps == cases[j].steps &&
                cases[i].width == cases[j].width && cases[i].height == cases[j].height &&
                cases[i].frames == cases[j].frames && cases[i].seed == cases[j].seed &&
                !strcmp(cases[i].prompt, cases[j].prompt)) { oracle = j; break; }
        }
        if (oracle >= 0) {
            if (!same(&cases[i].warm, &cases[oracle].warm)) {
                fprintf(stderr, "FAIL: restored state differs from its fresh oracle: %s\n", cases[i].name);
                ok = 0;
            }
            continue;
        }
        capture fresh;
        ctx = load_context(argv[1], argv[2], nrefs != 0);
        if (!ctx) { ok = 0; break; }
        size_t count = i == CASE_COUNT - 2 ? 1 : nrefs;
        ok = generate(ctx, &cases[i], dir, "fresh", refs, count, 0, &fresh, manifest);
        h3_free(ctx);
        if (ok && (!same(&cases[i].warm, &fresh) || fresh.loads != 1)) {
            fprintf(stderr, "FAIL: warm/fresh bytes or metadata differ: %s\n", cases[i].name);
            ok = 0;
        }
    }
    if (ok && (cases[0].warm.rgb == cases[1].warm.rgb ||
               cases[0].warm.rgb == cases[2].warm.rgb ||
               cases[2].warm.rgb == cases[3].warm.rgb)) {
        fprintf(stderr, "FAIL: changed seed/prompt/steps did not change RGB output\n");
        ok = 0;
    }
    if (ok && (!same(&cases[2].warm, &cases[4].warm) ||
               !same(&cases[2].warm, &cases[6].warm) ||
               !same(&cases[2].warm, &cases[7].warm) ||
               !same(&cases[2].warm, &cases[8].warm))) {
        fprintf(stderr, "FAIL: restored settings/clear/cancellation changed output\n");
        ok = 0;
    }
    if (fclose(manifest)) ok = 0;
    fprintf(stderr, "%s: real cache regression; manifest %s\n",
            ok ? "PASS" : "FAIL", manifest_path);
    free(dir);
    return ok ? 0 : 1;
}
