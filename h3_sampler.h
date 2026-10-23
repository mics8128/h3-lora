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
/* Called video first, then audio, once each per nonterminal LCM transition. */
typedef int (*h3_sampler_noise)(int step, int audio, float *noise,
    size_t count, void *opaque, char *error, size_t error_size);
/* Native states after each transition; returning zero cancels the run. */
typedef int (*h3_sampler_observe)(int completed, const float *video,
    const float *audio, void *opaque, char *error, size_t error_size);

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
