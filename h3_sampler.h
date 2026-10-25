#ifndef H3_SAMPLER_H
#define H3_SAMPLER_H

#include "h3.h"
#include "h3_host.h"
#include <stddef.h>
#include <stdint.h>

/* Joint native-state evaluation. Velocities point toward data: D=x+sigma*v. */
typedef int (*h3_sampler_velocity)(int step, const float *video,
    const float *audio, float *video_velocity, float *audio_velocity,
    void *opaque, char *error, size_t error_size);
/* Video first, then audio, once each per nonterminal LCM/ER-SDE transition. */
typedef int (*h3_sampler_noise)(int step, int audio, float *noise,
    size_t count, void *opaque, char *error, size_t error_size);
/* Native states after each transition; returning zero cancels the run. */
typedef int (*h3_sampler_observe)(int completed, const float *video,
    const float *audio, void *opaque, char *error, size_t error_size);

typedef struct {
    h3_sampler_kind kind;
    h3_sigma_schedule schedule;
    float video_shift, audio_shift;
    uint64_t seed;
    int start_step;
    size_t video_count, audio_count;
    /* Derive from effective conditioning, assets/model, shapes, execution
     * options and native implementation/backend fingerprint, not a user token.
     * Other config fields are additionally compared directly on load. */
    unsigned char request_digest[32];
} h3_sampler_config;
/* Apply Comfy's first-sigma CONST offset for ER-SDE before preparing model
 * timestep embeddings. Initial latent noising uses the original preoffset grid,
 * as in Comfy's outer sampler. State/run APIs require the effective grid. */
int h3_sampler_schedule_prepare(h3_sampler_kind, float video_shift,
    float audio_shift, h3_sigma_schedule *);
typedef struct h3_sampler_state h3_sampler_state;
/* All arrays are copied. Masks are per-element [0,1]: zero follows a fixed
 * clean/noise preservation trajectory, one generates freely. Each stream uses
 * its own sigma. Preservation noise is derived from the initial native latent
 * x=(1-sigma_start)*clean+sigma_start*noise; at terminal sigma zero it is clean.
 * Initial arrays are already at start_step, not scaled/noised here. A clean
 * array is required for each supplied mask. Both hard and soft masks project
 * before the first evaluation and after every transition. */
h3_sampler_state *h3_sampler_state_start(const h3_sampler_config *,
    const float *video, const float *audio,
    const float *video_clean, const float *video_mask,
    const float *audio_clean, const float *audio_mask, char *, size_t);
/* stop_step is an absolute completed-transition index. Callback cancellation
 * leaves a resumable completed boundary. External noise must be stateless by
 * step; its opaque state is NOT persisted (use NULL for native exact resume). */
int h3_sampler_state_run(h3_sampler_state *, int stop_step,
    h3_sampler_velocity, void *, h3_sampler_noise, void *,
    h3_sampler_observe, void *, char *, size_t);
int h3_sampler_state_save(const h3_sampler_state *, const char *, char *, size_t);
h3_sampler_state *h3_sampler_state_load(const char *, const h3_sampler_config *, char *, size_t);
/* Schema-validated diagnostic inspection without request identity verification.
 * Returned state permits getters/free only; run/save reject preview states. */
h3_sampler_state *h3_sampler_state_load_preview(const char *, char *, size_t);
void h3_sampler_state_free(h3_sampler_state *);
const float *h3_sampler_state_video(const h3_sampler_state *);
const float *h3_sampler_state_audio(const h3_sampler_state *);
int h3_sampler_state_step(const h3_sampler_state *);
const h3_sampler_config *h3_sampler_state_config(const h3_sampler_state *);

/* CPU orchestration only; the callback may execute the model on Metal.
 * With noise=NULL, run-local SplitMix64/Box-Muller Gaussian noise is used.
 * Its seed is reproducible natively, not PyTorch-identical. No RNG/history is
 * retained between calls. Initial inputs are native, unscaled latents. */
int h3_sampler_run(h3_sampler_kind kind, const h3_sigma_schedule *schedule,
    float video_shift, float audio_shift, uint64_t seed,
    float *video, size_t video_count, float *audio, size_t audio_count,
    h3_sampler_velocity evaluate, void *evaluate_opaque,
    h3_sampler_noise noise, void *noise_opaque,
    h3_sampler_observe observe, void *observe_opaque,
    char *error, size_t error_size);
#endif
