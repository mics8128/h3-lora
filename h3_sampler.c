#include "h3_sampler.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Independent implementation of flow/LCM updates and packed AV coordinates.
 * LCM: https://arxiv.org/abs/2310.04378
 * RES: https://arxiv.org/abs/2308.02157 (h3_res_step numerical primitive).
 * AV coordinate contract: ComfyUI bc5f880dbdd43e426077f42ccec53bb577ffee9f,
 * ModelSamplingAV; no upstream implementation code is incorporated here. */
static int reject(char *error, size_t size, const char *message) {
    if (error && size) snprintf(error, size, "%s", message);
    return 0;
}

static uint64_t random_bits(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static void gaussian(float *values, size_t count, uint64_t *state) {
    for (size_t i = 0; i < count; i += 2) {
        double u = ((double)(random_bits(state) >> 11) + 0.5) / 9007199254740992.0;
        double v = ((double)(random_bits(state) >> 11) + 0.5) / 9007199254740992.0;
        double radius = sqrt(-2.0 * log(u));
        double angle = 6.2831853071795864769 * v;
        values[i] = (float)(radius * cos(angle));
        if (i + 1 < count) values[i + 1] = (float)(radius * sin(angle));
    }
}

int h3_sampler_run(h3_sampler_kind kind, const h3_sigma_schedule *schedule,
    float video_shift, float audio_shift, uint64_t seed,
    float *video, size_t video_count, float *audio, size_t audio_count,
    h3_sampler_velocity evaluate, void *evaluate_opaque,
    h3_sampler_noise noise, void *noise_opaque,
    h3_sampler_observe observe, void *observe_opaque,
    char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!schedule || schedule->steps < 1 || schedule->steps > H3_MAX_STEPS ||
        !video || !audio || !video_count || !audio_count || !evaluate ||
        !isfinite(video_shift) || !isfinite(audio_shift) ||
        video_shift <= 0 || audio_shift <= 0 ||
        kind < H3_SAMPLER_EULER || kind > H3_SAMPLER_RES_MULTISTEP)
        return reject(error, error_size, "invalid simple sampler arguments");
    for (int i = 0; i < schedule->steps; i++) {
        if (!isfinite(schedule->video[i]) || !isfinite(schedule->audio[i]) ||
            !(schedule->video[i] > schedule->video[i + 1]) ||
            !(schedule->audio[i] > schedule->audio[i + 1]) ||
            schedule->video[i + 1] < 0 || schedule->audio[i + 1] < 0)
            return reject(error, error_size, "invalid simple sampler sigma grid");
    }
    if (schedule->video[schedule->steps] != 0 ||
        schedule->audio[schedule->steps] != 0)
        return reject(error, error_size, "simple sampler requires terminal zero");
    float c = video_shift / audio_shift;
    if (!isfinite(c) || c <= 0)
        return reject(error, error_size, "invalid audiovisual shift ratio");
    size_t slots = kind == H3_SAMPLER_RES_MULTISTEP ? 4 : 3;
    if (video_count > SIZE_MAX - audio_count ||
        video_count + audio_count > SIZE_MAX / (slots * sizeof(float)))
        return reject(error, error_size, "sampler state is too large");
    size_t count = video_count + audio_count;
    float *workspace = malloc(slots * count * sizeof(float));
    if (!workspace) return reject(error, error_size, "out of memory allocating sampler state");
    float *carried = workspace, *velocity = carried + count;
    float *clean = velocity + count, *old = clean + count;
    /* Velocity is dead once clean predictions exist; reuse it for LCM noise. */
    float *draw = velocity;
    memcpy(carried, video, video_count * sizeof(float));
    float r = c + (1.0f - c) * schedule->video[0];
    for (size_t j = 0; j < audio_count; j++) carried[video_count + j] = r * audio[j];
    int ok = 1;
    for (int step = 0; step < schedule->steps && ok; step++) {
        float sigma = schedule->video[step], next = schedule->video[step + 1];
        r = c + (1.0f - c) * sigma;
        memcpy(video, carried, video_count * sizeof(float));
        for (size_t j = 0; j < audio_count; j++) audio[j] = carried[video_count + j] / r;
        ok = evaluate(step, video, audio, velocity, velocity + video_count,
                      evaluate_opaque, error, error_size);
        if (!ok) break;
        for (size_t j = 0; j < video_count; j++) clean[j] = video[j] + sigma * velocity[j];
        for (size_t j = 0; j < audio_count; j++)
            clean[video_count + j] = c * (audio[j] + schedule->audio[step] * velocity[video_count + j]);
        if (kind == H3_SAMPLER_RES_MULTISTEP) {
            ok = h3_res_step(carried, carried, clean, step ? old : NULL,
                             count, schedule->video, step, schedule->steps);
            if (!ok) reject(error, error_size, "RES solver rejected transition");
            memcpy(old, clean, count * sizeof(float));
        } else if (kind == H3_SAMPLER_LCM) {
            if (next != 0) {
                if (noise) {
                    ok = noise(step, 0, draw, video_count, noise_opaque, error, error_size) &&
                         noise(step, 1, draw + video_count, audio_count, noise_opaque, error, error_size);
                } else {
                    gaussian(draw, video_count, &seed);
                    gaussian(draw + video_count, audio_count, &seed);
                }
                if (ok) for (size_t j = 0; j < count; j++)
                    carried[j] = (1.0f - next) * clean[j] + next * draw[j];
            } else memcpy(carried, clean, count * sizeof(float));
        } else {
            if (next == 0) memcpy(carried, clean, count * sizeof(float));
            else for (size_t j = 0; j < count; j++)
                carried[j] += (next - sigma) * ((carried[j] - clean[j]) / sigma);
        }
        if (!ok) break;
        memcpy(video, carried, video_count * sizeof(float));
        float next_r = c + (1.0f - c) * next;
        for (size_t j = 0; j < audio_count; j++) audio[j] = carried[video_count + j] / next_r;
        if (observe) ok = observe(step + 1, video, audio, observe_opaque, error, error_size);
    }
    free(workspace);
    return ok;
}
