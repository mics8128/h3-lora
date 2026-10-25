#include "h3_sampler.h"
#include "h3_latent.h"
#include <unistd.h>

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


static int zero_velocity(int step, const float *video, const float *audio,
    float *vv, float *av, void *opaque, char *error, size_t error_size) {
    (void)step; (void)video; (void)audio; (void)opaque;
    (void)error; (void)error_size;
    vv[0]=vv[1]=0;
    av[0]=av[1]=av[2]=0;
    return 1;
}

static void test_soft_mask_single_projection(void) {
    char error[256];
    for (int kind=H3_SAMPLER_EULER;kind<=H3_SAMPLER_RES_MULTISTEP;kind++) {
        h3_sampler_config c={0};c.kind=(h3_sampler_kind)kind;
        c.video_shift=1;c.audio_shift=1;c.video_count=2;c.audio_count=3;
        CHECK(h3_schedule_build(1,1,1,&c.schedule));
        float video[]={0,0},audio[]={0,0,0};
        float clean_video[]={10,10},clean_audio[]={10,10,10};
        float video_mask[]={.5f,.5f},audio_mask[]={.5f,.5f,.5f};
        h3_sampler_state *s=h3_sampler_state_start(&c,video,audio,
            clean_video,video_mask,clean_audio,audio_mask,error,sizeof(error));
        CHECK(s);
        CHECK(h3_sampler_state_run(s,1,zero_velocity,NULL,NULL,NULL,NULL,NULL,error,sizeof(error)));
        /* Initial zero plus zero velocity proposes zero. A single terminal
         * 50% preservation projection against clean ten must produce five,
         * not 7.5 from separately masking the model prediction and proposal. */
        for (int j=0;j<2;j++)CHECK(h3_sampler_state_video(s)[j]==5);
        for (int j=0;j<3;j++)CHECK(h3_sampler_state_audio(s)[j]==5);
        h3_sampler_state_free(s);
    }
}

static void test_preservation_trajectory(void) {
    char error[256];
    for(int kind=H3_SAMPLER_EULER;kind<=H3_SAMPLER_RES_MULTISTEP;kind++){
        h3_sampler_config c={0};c.kind=(h3_sampler_kind)kind;
        c.video_shift=3;c.audio_shift=1;c.video_count=2;c.audio_count=3;
        c.schedule.steps=2;c.schedule.video[0]=1;c.schedule.video[1]=.5f;
        c.schedule.audio[0]=1;c.schedule.audio[1]=.25f;
        float v[]={10,12},a[]={14,16,18},vc[]={2,4},ac[]={6,8,10};
        float vm[]={0,0},am[]={0,0,0};
        h3_sampler_state *s=h3_sampler_state_start(&c,v,a,vc,vm,ac,am,error,sizeof(error));
        CHECK(s);
        CHECK(h3_sampler_state_video(s)[0]==10&&h3_sampler_state_audio(s)[0]==14);
        CHECK(h3_sampler_state_run(s,1,native_model,NULL,NULL,NULL,NULL,NULL,error,sizeof(error)));
        /* Fixed noise remains 10/12 and 14/16/18. Each modality follows its
         * own sigma: halfway video, one-quarter audio, not a clean clamp. */
        close_value(h3_sampler_state_video(s)[0],6);
        close_value(h3_sampler_state_video(s)[1],8);
        close_value(h3_sampler_state_audio(s)[0],8);
        close_value(h3_sampler_state_audio(s)[1],10);
        close_value(h3_sampler_state_audio(s)[2],12);
        CHECK(h3_sampler_state_run(s,2,native_model,NULL,NULL,NULL,NULL,NULL,error,sizeof(error)));
        CHECK(!memcmp(h3_sampler_state_video(s),vc,sizeof(vc)));
        CHECK(!memcmp(h3_sampler_state_audio(s),ac,sizeof(ac)));
        h3_sampler_state_free(s);
    }
}

static void test_state_resume(void) {
    char error[256], path[128];
    snprintf(path, sizeof(path), "/tmp/h3-state-%ld.bin", (long)getpid());
    for (int kind = H3_SAMPLER_EULER; kind <= H3_SAMPLER_ER_SDE; kind++) {
        for (int start = 0; start <= 2; start += 2) {
            h3_sampler_config c = {0};
            c.kind = (h3_sampler_kind)kind; c.video_shift = 12; c.audio_shift = 2;
            c.video_count = 2; c.audio_count = 3; c.seed = 456; c.start_step = start;
            c.request_digest[0] = 42;
            CHECK(h3_schedule_build(6, 12, 2, &c.schedule));
            CHECK(h3_sampler_schedule_prepare(c.kind,12,2,&c.schedule));
            float v[] = {.3f, -.7f}, a[] = {.2f, -.4f, .8f};
            float vc[] = {7, 8}, ac[] = {9, 10, 11};
            float vm[] = {0, 1}, am[] = {1, 0, .5f};
            h3_sampler_state *full = h3_sampler_state_start(&c,v,a,vc,vm,ac,am,error,sizeof(error));
            CHECK(full);
            CHECK(h3_sampler_state_run(full,6,native_model,NULL,NULL,NULL,NULL,NULL,error,sizeof(error)));
            for (int split = start; split <= 6; split++) {
                h3_sampler_state *part = h3_sampler_state_start(&c,v,a,vc,vm,ac,am,error,sizeof(error));
                CHECK(part);
                CHECK(h3_sampler_state_run(part,split,native_model,NULL,NULL,NULL,NULL,NULL,error,sizeof(error)));
                if(split==6){CHECK(h3_sampler_state_video(part)[0] == 7);CHECK(h3_sampler_state_audio(part)[1] == 10);}
                CHECK(h3_sampler_state_save(part,path,error,sizeof(error)));
                h3_sampler_state *preview=h3_sampler_state_load_preview(path,error,sizeof(error));
                CHECK(preview && h3_sampler_state_step(preview)==split);
                CHECK(!memcmp(h3_sampler_state_video(preview),h3_sampler_state_video(part),2*sizeof(float)));
                CHECK(!memcmp(h3_sampler_state_audio(preview),h3_sampler_state_audio(part),3*sizeof(float)));
                CHECK(!h3_sampler_state_run(preview,6,native_model,NULL,NULL,NULL,NULL,NULL,error,sizeof(error)));
                CHECK(!h3_sampler_state_save(preview,path,error,sizeof(error)));
                h3_sampler_state_free(preview);
                CHECK(!h3_sampler_state_load(path,NULL,error,sizeof(error)));
                h3_sampler_state_free(part);
                if(c.kind==H3_SAMPLER_ER_SDE) {
                    void *payload=NULL;size_t bytes=0;
                    CHECK(h3_native_read(path,4,&payload,&bytes,error,sizeof(error)));
                    CHECK(bytes==136+8*(size_t)(c.schedule.steps+1)+28*(c.video_count+c.audio_count));
                    uint64_t *header=payload;CHECK(header[10]==1);
                    header[10]=0;
                    CHECK(h3_native_write(path,4,payload,bytes,error,sizeof(error)));
                    CHECK(!h3_sampler_state_load(path,&c,error,sizeof(error)));
                    CHECK(!h3_sampler_state_load_preview(path,error,sizeof(error)));
                    header[10]=2;
                    CHECK(h3_native_write(path,4,payload,bytes,error,sizeof(error)));
                    CHECK(!h3_sampler_state_load(path,&c,error,sizeof(error)));
                    header[10]=1;
                    CHECK(h3_native_write(path,4,payload,bytes-4,error,sizeof(error)));
                    CHECK(!h3_sampler_state_load(path,&c,error,sizeof(error)));
                    CHECK(h3_native_write(path,4,payload,bytes,error,sizeof(error)));
                    free(payload);
                }
                part = h3_sampler_state_load(path,&c,error,sizeof(error));
                CHECK(part && h3_sampler_state_step(part) == split);
                CHECK(h3_sampler_state_run(part,6,native_model,NULL,NULL,NULL,NULL,NULL,error,sizeof(error)));
                CHECK(!memcmp(h3_sampler_state_video(part),h3_sampler_state_video(full),2*sizeof(float)));
                CHECK(!memcmp(h3_sampler_state_audio(part),h3_sampler_state_audio(full),3*sizeof(float)));
                h3_sampler_state_free(part);
                c.request_digest[1]++;
                CHECK(!h3_sampler_state_load(path,&c,error,sizeof(error)));
                c.request_digest[1]--;
                c.seed++;
                CHECK(!h3_sampler_state_load(path,&c,error,sizeof(error)));
                c.seed--;
                c.schedule.audio[1] *= .99f;
                CHECK(!h3_sampler_state_load(path,&c,error,sizeof(error)));
                CHECK(h3_schedule_build(6,12,2,&c.schedule));
                CHECK(h3_sampler_schedule_prepare(c.kind,12,2,&c.schedule));
                FILE *f = fopen(path,"wb"); CHECK(f); CHECK(fwrite("H3NATIVE",1,8,f)==8); CHECK(!fclose(f));
                CHECK(!h3_sampler_state_load(path,&c,error,sizeof(error)));
                CHECK(!h3_sampler_state_load_preview(path,error,sizeof(error)));
            }
            vm[1] = 1.01f;
            CHECK(!h3_sampler_state_start(&c,v,a,vc,vm,ac,am,error,sizeof(error)));
            vm[1] = NAN;
            CHECK(!h3_sampler_state_start(&c,v,a,vc,vm,ac,am,error,sizeof(error)));
            h3_sampler_state_free(full);
        }
    }
    unlink(path);
}

static void test_native_latent_file(void) {
    char error[256], path[128];
    snprintf(path,sizeof(path),"/tmp/h3-latent-%ld.bin",(long)getpid());
    float v[168] = {1,2}, a[64] = {3,4,5};
    h3_latent input = {7,1,1,1,168,64,v,a,22,H3_LATENT_TEMPORAL_ENCODER_LINEAR_ENDPOINTS}, output = {0};
    CHECK(h3_latent_save(path,&input,error,sizeof(error)));
    CHECK(h3_latent_load(path,&output,error,sizeof(error)));
    CHECK(output.video_count == 168 && output.audio_count == 64);
    CHECK(output.source_frames==22 && output.temporal_mapping==1);
    CHECK(!memcmp(v,output.video,sizeof(v)) && !memcmp(a,output.audio,sizeof(a)));
    h3_latent_free(&output);
    /* Shape remains internally valid but provenance must not claim that a
     * six-step raw encoder latent is a seven-step generated-cadence latent. */
    input.video_frames=6; input.video_count=144;
    CHECK(!h3_latent_save(path,&input,error,sizeof(error)));
    input.video_frames=7; input.video_count=168;
    input.source_frames=5;
    CHECK(!h3_latent_save(path,&input,error,sizeof(error)));
    input.source_frames=0;
    CHECK(!h3_latent_save(path,&input,error,sizeof(error)));
    input.source_frames=22; input.temporal_mapping=H3_LATENT_TEMPORAL_NATIVE;
    CHECK(h3_latent_save(path,&input,error,sizeof(error)));
    input.source_frames=5;
    CHECK(!h3_latent_save(path,&input,error,sizeof(error)));
    input.source_frames=22; input.temporal_mapping=H3_LATENT_TEMPORAL_ENCODER_LINEAR_ENDPOINTS;
    input.height = 0;
    CHECK(!h3_latent_save(path,&input,error,sizeof(error)));
    input.height = 1; v[0] = INFINITY;
    CHECK(!h3_latent_save(path,&input,error,sizeof(error)));
    FILE *f = fopen(path,"r+b"); CHECK(f); CHECK(!fseek(f,-1,SEEK_END)); CHECK(fputc(255,f)!=EOF); CHECK(!fclose(f));
    CHECK(!h3_latent_load(path,&output,error,sizeof(error)));
    v[0] = 1;
    input.video_count = 23;
    CHECK(!h3_latent_save(path,&input,error,sizeof(error)));
    input.video_count = 0; input.video_frames = input.height = input.width = 0;
    input.source_frames=0; input.temporal_mapping=0;
    input.video = NULL;
    CHECK(h3_latent_save(path,&input,error,sizeof(error)));
    CHECK(h3_latent_load(path,&output,error,sizeof(error)));
    CHECK(output.video_count == 0 && output.video == NULL && output.audio_count == 64);
    h3_latent_free(&output);
    unlink(path);
}

/* Literal scalar reference vectors from ComfyUI sample_er_sde (revision
 * fe298e4e20f1ff26d30e8639267bbd0b57811390), CONST, max_stage=3,
 * default phi and 200 left nodes, evaluated independently in double.
 * Deliberately nonuniform sigmas and changing predictions exercise both
 * clean-prediction and derivative history, not just constant-model Euler. */
static const float er_predictions[4][2]={{.2f,.5f},{.7f,-.1f},{-.2f,.8f},{.4f,.3f}};
static int er_reference_model(int step,const float *v,const float *a,
    float *dv,float *da,void *opaque,char *error,size_t size) {
    (void)error;(void)size;
    const float sigmas[]={.9f,.65f,.35f,.1f};
    dv[0]=(er_predictions[step][0]-v[0])/sigmas[step];
    const h3_sampler_config *c=opaque;
    da[0]=(er_predictions[step][1]-a[0])/(c?c->schedule.audio[step]:sigmas[step]);
    return 1;
}
static int er_reference_noise(int step,int audio,float *out,size_t count,
    void *opaque,char *error,size_t size) {
    (void)error;(void)size;
    CHECK(count==1);CHECK(step<3);
    out[0]=opaque ? 0 : (audio?-.3f:.1f)*(float)(step+1);
    return 1;
}
static void test_er_reference_vectors(void) {
    const long double expected[4][2]={
        {.269422328631120L,-.3720183848322928L},
        {.5248428050446213L,-.3503557219110296L},
        {-.1567120019773084L,.6195909599545141L},{.4L,.3L}};
    h3_sampler_config c={0};c.kind=H3_SAMPLER_ER_SDE;
    c.video_shift=c.audio_shift=1;c.video_count=c.audio_count=1;c.schedule.steps=4;
    float grid[]={.9f,.65f,.35f,.1f,0};
    memcpy(c.schedule.video,grid,sizeof(grid));memcpy(c.schedule.audio,grid,sizeof(grid));
    float video=.3f,audio=-.7f;char error[256],path[128];
    h3_sampler_state *s=h3_sampler_state_start(&c,&video,&audio,NULL,NULL,NULL,NULL,error,sizeof(error));
    CHECK(s);
    for(int stop=1;stop<=4;stop++) {
        CHECK(h3_sampler_state_run(s,stop,er_reference_model,NULL,er_reference_noise,NULL,NULL,NULL,error,sizeof(error)));
        close_value(h3_sampler_state_video(s)[0],expected[stop-1][0]);
        close_value(h3_sampler_state_audio(s)[0],expected[stop-1][1]);
    }
    snprintf(path,sizeof(path),"/tmp/h3-er-external-%ld.bin",(long)getpid());
    CHECK(!h3_sampler_state_save(s,path,error,sizeof(error)));
    h3_sampler_state_free(s);
    const long double zero_noise[4][2]={
        {.22944005911188503L,-.25207157627458776L},
        {.4778990847093809L,-.2095245609053084L},
        {-.17883597024687342L,.685962864763209L},{.4L,.3L}};
    s=h3_sampler_state_start(&c,&video,&audio,NULL,NULL,NULL,NULL,error,sizeof(error));CHECK(s);
    for(int stop=1;stop<=4;stop++) {
        CHECK(h3_sampler_state_run(s,stop,er_reference_model,NULL,er_reference_noise,&c,NULL,NULL,error,sizeof(error)));
        close_value(h3_sampler_state_video(s)[0],zero_noise[stop-1][0]);
        close_value(h3_sampler_state_audio(s)[0],zero_noise[stop-1][1]);
    }
    h3_sampler_state_free(s);
    /* Unequal shifts: upstream solves the single carried pack, with Dhat_a
     * =3*D_a, then unpacks at each boundary; per-stream ER is not equivalent. */
    const long double packed_audio[]={-.0933419383369925L,-.17964301448040268L,
        .7457812398367875L,.3L};
    c.video_shift=3;
    for(int i=0;i<=4;i++)c.schedule.audio[i]=grid[i]/(3-2*grid[i]);
    s=h3_sampler_state_start(&c,&video,&audio,NULL,NULL,NULL,NULL,error,sizeof(error));CHECK(s);
    for(int stop=1;stop<=4;stop++) {
        CHECK(h3_sampler_state_run(s,stop,er_reference_model,&c,er_reference_noise,NULL,NULL,NULL,error,sizeof(error)));
        close_value(h3_sampler_state_video(s)[0],expected[stop-1][0]);
        close_value(h3_sampler_state_audio(s)[0],packed_audio[stop-1]);
    }
    h3_sampler_state_free(s);
    CHECK(h3_schedule_build(4,12,3,&c.schedule));
    CHECK(!h3_sampler_state_start(&c,&video,&audio,NULL,NULL,NULL,NULL,error,sizeof(error)));
    CHECK(h3_sampler_schedule_prepare(c.kind,12,3,&c.schedule));
    CHECK(c.schedule.video[0]<1 && c.schedule.audio[0]<1);
    close_value(c.schedule.video[0],12.L*.9999L/(1+11.L*.9999L));
    close_value(c.schedule.audio[0],3.L*.9999L/(1+2.L*.9999L));
}

int main(void) {
    test_soft_mask_single_projection();
    test_preservation_trajectory();
    test_state_resume();
    test_native_latent_file();
    test_er_reference_vectors();
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
