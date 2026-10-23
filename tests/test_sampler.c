#include "h3_sampler.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

typedef struct {
    h3_sampler_kind kind;
    h3_sigma_schedule schedule;
    float video_shift, audio_shift;
    int evaluations, draws, observations, cancel;
    long double carried[5], old[5];
} oracle;

static void close_value(float actual, long double expected) {
    CHECK(isfinite(actual));
    CHECK(fabsl(actual - expected) < 4e-5L * (1 + fabsl(expected)));
}

/* Coupled affine model: each stream changes the other's prediction. */
static void model(const long double *video, const long double *audio,
                  long double *velocity) {
    velocity[0] = 0.17L * video[0] - 0.11L * audio[0] + 0.03L;
    velocity[1] = -0.13L * video[1] + 0.07L * audio[2] - 0.02L;
    velocity[2] = 0.09L * video[0] - 0.19L * audio[0] + 0.04L;
    velocity[3] = -0.06L * video[1] + 0.14L * audio[1] - 0.01L;
    velocity[4] = 0.05L * (video[0] + video[1]) - 0.12L * audio[2];
}

static float noise_value(int step, int audio, size_t index) {
    return (float)((step + 1) * 0.03125 + (audio ? -0.25 : 0.125) + (double)index * 0.0625);
}

static int supplied_noise(int step, int audio, float *out, size_t count,
    void *opaque, char *error, size_t error_size) {
    (void)error; (void)error_size;
    oracle *o = opaque;
    CHECK(step < o->schedule.steps - 1);
    CHECK(audio == o->draws % 2);
    CHECK(count == (audio ? 3u : 2u));
    o->draws++;
    for (size_t j = 0; j < count; j++) out[j] = noise_value(step, audio, j);
    return 1;
}

/* Independent oracle: integrate an affine-in-log-time clean predictor exactly.
 * Its exponential zeroth/first moments are used directly, not phi helpers or
 * h3_res_step. These are the RES eta=0 polynomial quadrature moments. */
static void oracle_transition(oracle *o, int step, const long double *native) {
    long double velocity[5], clean[5];
    model(native, native + 2, velocity);
    long double sigma = o->schedule.video[step];
    long double next = o->schedule.video[step + 1];
    long double c = (float)(o->video_shift / o->audio_shift);
    for (int j = 0; j < 5; j++) {
        long double stream_sigma = j < 2 ? sigma : o->schedule.audio[step];
        clean[j] = (native[j] + stream_sigma * velocity[j]) * (j < 2 ? 1 : c);
    }
    for (int j = 0; j < 5; j++) {
        long double current = o->carried[j];
        if (o->kind == H3_SAMPLER_LCM) {
            o->carried[j] = (1 - next) * clean[j];
            if (next != 0) o->carried[j] += next * noise_value(step, j >= 2, j < 2 ? (size_t)j : (size_t)(j - 2));
        } else if (o->kind == H3_SAMPLER_RES_MULTISTEP && step && next != 0) {
            long double h = logl(sigma / next);
            long double previous_h = logl(o->schedule.video[step - 1] / sigma);
            long double decay = next / sigma;
            long double moment0 = 1 - decay;
            long double moment1 = h - moment0;
            long double slope = (clean[j] - o->old[j]) / previous_h;
            o->carried[j] = decay * current + moment0 * clean[j] + moment1 * slope;
        } else o->carried[j] = next != 0 ? current * (next / sigma) + (1 - next / sigma) * clean[j] : clean[j];
        o->old[j] = clean[j];
    }
}

static int evaluate(int step, const float *video, const float *audio,
    float *vv, float *av, void *opaque, char *error, size_t error_size) {
    (void)error; (void)error_size;
    oracle *o = opaque;
    CHECK(step == o->evaluations++);
    long double native[5], velocity[5];
    float c = o->video_shift / o->audio_shift;
    float r = c + (1.0f - c) * o->schedule.video[step];
    for (int j = 0; j < 5; j++) {
        native[j] = o->carried[j] / (j < 2 ? 1 : r);
        close_value(j < 2 ? video[j] : audio[j - 2], native[j]);
    }
    model(native, native + 2, velocity);
    vv[0] = (float)velocity[0]; vv[1] = (float)velocity[1];
    for (int j = 0; j < 3; j++) av[j] = (float)velocity[j + 2];
    oracle_transition(o, step, native);
    return 1;
}

static int observe(int completed, const float *video, const float *audio,
    void *opaque, char *error, size_t error_size) {
    oracle *o = opaque;
    CHECK(completed == ++o->observations);
    float c = o->video_shift / o->audio_shift;
    float r = c + (1.0f - c) * o->schedule.video[completed];
    for (int j = 0; j < 2; j++) close_value(video[j], o->carried[j]);
    for (int j = 0; j < 3; j++) close_value(audio[j], o->carried[j + 2] / r);
    if (o->cancel) {
        snprintf(error, error_size, "test cancellation");
        return 0;
    }
    return 1;
}

static oracle make_oracle(h3_sampler_kind kind, int steps, float vs, float as) {
    oracle o = {0};
    o.kind = kind; o.video_shift = vs; o.audio_shift = as;
    CHECK(h3_schedule_build(steps, vs, as, &o.schedule));
    long double initial[] = {0.3L, -0.7L, 0.2L, -0.4L, 0.8L};
    memcpy(o.carried, initial, sizeof(initial));
    return o;
}

static void run_case(h3_sampler_kind kind, int steps, float vs, float as) {
    oracle o = make_oracle(kind, steps, vs, as);
    float video[] = {0.3f, -0.7f}, audio[] = {0.2f, -0.4f, 0.8f};
    char error[256];
    CHECK(h3_sampler_run(kind, &o.schedule, vs, as, 123,
        video, 2, audio, 3, evaluate, &o, supplied_noise, &o,
        observe, &o, error, sizeof(error)));
    CHECK(o.evaluations == steps && o.observations == steps);
    CHECK(o.draws == (kind == H3_SAMPLER_LCM ? 2 * (steps - 1) : 0));
    /* Fresh history and injected noise must make a second run identical. */
    oracle again = make_oracle(kind, steps, vs, as);
    float v2[] = {0.3f, -0.7f}, a2[] = {0.2f, -0.4f, 0.8f};
    CHECK(h3_sampler_run(kind, &again.schedule, vs, as, 999,
        v2, 2, a2, 3, evaluate, &again, supplied_noise, &again,
        observe, &again, error, sizeof(error)));
    CHECK(!memcmp(video, v2, sizeof(video)) && !memcmp(audio, a2, sizeof(audio)));
}

static int native_model(int step, const float *v, const float *a, float *vv,
    float *av, void *opaque, char *error, size_t error_size) {
    (void)step; (void)opaque; (void)error; (void)error_size;
    long double video[] = {v[0], v[1]}, audio[] = {a[0], a[1], a[2]}, velocity[5];
    model(video, audio, velocity);
    vv[0] = (float)velocity[0]; vv[1] = (float)velocity[1];
    for (int j = 0; j < 3; j++) av[j] = (float)velocity[j + 2];
    return 1;
}

static void test_seeded_noise(void) {
    h3_sigma_schedule schedule;
    CHECK(h3_schedule_build(6, 12, 2, &schedule));
    float result[3][5];
    for (int run = 0; run < 3; run++) {
        float initial[] = {0.3f, -0.7f, 0.2f, -0.4f, 0.8f};
        memcpy(result[run], initial, sizeof(initial));
        char error[256];
        CHECK(h3_sampler_run(H3_SAMPLER_LCM, &schedule, 12, 2,
            run == 2 ? 457 : 456, result[run], 2, result[run] + 2, 3,
            native_model, NULL, NULL, NULL, NULL, NULL, error, sizeof(error)));
    }
    CHECK(!memcmp(result[0], result[1], sizeof(result[0])));
    CHECK(memcmp(result[0], result[2], sizeof(result[0])));
}

static void test_res_worked_example(void) {
    float sigmas[] = {1, 0.5f, 0.25f, 0}, sample[] = {2}, clean[] = {3}, old[] = {1}, out[1];
    CHECK(h3_res_step(out, sample, clean, old, 1, sigmas, 1, 3));
    /* Exact quadrature: 1 + 1.5 + 2*(log(2)-0.5)/log(2). */
    close_value(out[0], 3.0573049591110366L);
    CHECK(h3_res_step(out, sample, clean, NULL, 1, sigmas, 0, 3));
    close_value(out[0], 2.5L);
    CHECK(h3_res_step(out, sample, clean, old, 1, sigmas, 2, 3));
    close_value(out[0], 3.0L);
}

static void test_six_step_schedule(void) {
    h3_sigma_schedule simple, serving;
    CHECK(h3_schedule_build(6, 12, 3, &simple));
    CHECK(h3_serving_schedule_build(6, 12, 3, &serving));
    /* Discrete entry 834/1000, not continuous 5/6. */
    close_value(simple.video[1], 0.9836839001376056L);
    close_value(simple.audio[1], 0.9377811094452774L);
    CHECK(simple.video[1] != serving.video[1]);
    CHECK(simple.audio[1] != serving.audio[1]);
}

static void test_cancel(void) {
    oracle o = make_oracle(H3_SAMPLER_LCM, 4, 12, 3);
    o.cancel = 1;
    float v[] = {0.3f, -0.7f}, a[] = {0.2f, -0.4f, 0.8f};
    char error[256];
    CHECK(!h3_sampler_run(o.kind, &o.schedule, 12, 3, 1, v, 2, a, 3,
        evaluate, &o, supplied_noise, &o, observe, &o, error, sizeof(error)));
    CHECK(o.evaluations == 1 && o.observations == 1 && o.draws == 2);
}
static void test_one_step_clean(void) {
    h3_sigma_schedule schedule;
    CHECK(h3_schedule_build(1, 12, 2, &schedule));
    const long double expected[] = {0.359L, -0.573L, 0.229L, -0.424L, 0.684L};
    for (int kind = H3_SAMPLER_EULER; kind <= H3_SAMPLER_RES_MULTISTEP; kind++) {
        float state[] = {0.3f, -0.7f, 0.2f, -0.4f, 0.8f};
        char error[256];
        CHECK(h3_sampler_run((h3_sampler_kind)kind, &schedule, 12, 2, 0,
            state, 2, state + 2, 3, native_model, NULL, NULL, NULL,
            NULL, NULL, error, sizeof(error)));
        for (int j = 0; j < 5; j++) close_value(state[j], expected[j]);
    }
}


int main(void) {
    const int steps[] = {1, 4, 5, 6, 8};
    const float shifts[][2] = {{12, 3}, {12, 2}, {6, 3}};
    test_res_worked_example();
    test_six_step_schedule();
    test_one_step_clean();
    for (int kind = H3_SAMPLER_EULER; kind <= H3_SAMPLER_RES_MULTISTEP; kind++)
        for (size_t n = 0; n < sizeof(steps) / sizeof(steps[0]); n++)
            for (size_t s = 0; s < sizeof(shifts) / sizeof(shifts[0]); s++)
                run_case((h3_sampler_kind)kind, steps[n], shifts[s][0], shifts[s][1]);
    test_seeded_noise();
    test_cancel();
    puts("sampler tests passed");
    return 0;
}
