#include "h3_sampler.h"
#include "h3_latent.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int reject(char *e, size_t n, const char *m) {
    if (e && n)
        snprintf(e, n, "%s", m);
    return 0;
}
static uint64_t random_bits(uint64_t *s) {
    uint64_t z = (*s += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
static void gaussian(float *v, size_t n, uint64_t *s) {
    for (size_t i = 0; i < n; i += 2) {
        double u = ((double)(random_bits(s) >> 11) + 0.5) / 9007199254740992.0,
               w = ((double)(random_bits(s) >> 11) + 0.5) / 9007199254740992.0,
               r = sqrt(-2 * log(u)), a = 6.2831853071795864769 * w;
        v[i] = (float)(r * cos(a));
        if (i + 1 < n)
            v[i + 1] = (float)(r * sin(a));
    }
}
struct h3_sampler_state {
    h3_sampler_config config;
    int step, history, external_noise, preview_only;
    uint64_t rng;
    size_t count;
    float *storage, *mask_storage, *native, *carried, *velocity, *prediction, *old, *anchor, *mask,
        *preserve_noise, *old_derivative;
};
static int finite_values(const float *p, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (!isfinite(p[i]))
            return 0;
    return 1;
}
static int valid_config(const h3_sampler_config *c) {
    if (!c || c->kind < H3_SAMPLER_EULER || c->kind > H3_SAMPLER_ER_SDE ||
        c->schedule.steps < 1 || c->schedule.steps > H3_MAX_STEPS || c->start_step < 0 ||
        c->start_step > c->schedule.steps || !c->video_count || !c->audio_count ||
        c->video_count > SIZE_MAX - c->audio_count ||
        (c->video_count + c->audio_count) > SIZE_MAX / (8 * sizeof(float)) ||
        !isfinite(c->video_shift) || !isfinite(c->audio_shift) || c->video_shift <= 0 ||
        c->audio_shift <= 0 || !isfinite(c->video_shift / c->audio_shift) ||
        c->video_shift / c->audio_shift <= 0)
        return 0;
    for (int i = 0; i < c->schedule.steps; i++)
        if (!isfinite(c->schedule.video[i]) || !isfinite(c->schedule.audio[i]) ||
            !(c->schedule.video[i] > c->schedule.video[i + 1]) ||
            !(c->schedule.audio[i] > c->schedule.audio[i + 1]) || c->schedule.video[i + 1] < 0 ||
            c->schedule.audio[i + 1] < 0)
            return 0;
    if (c->kind == H3_SAMPLER_ER_SDE &&
        (c->schedule.video[0] >= 1 || c->schedule.audio[0] >= 1)) return 0;
    float ratio = c->video_shift / c->audio_shift;
    for (int i = 0; i <= c->schedule.steps; i++) {
        float r = ratio + (1 - ratio) * c->schedule.video[i];
        if (!isfinite(r) || r <= 0)
            return 0;
    }
    return c->schedule.video[c->schedule.steps] == 0 && c->schedule.audio[c->schedule.steps] == 0;
}
static h3_sampler_state *allocate(const h3_sampler_config *c, int masks) {
    h3_sampler_state *s = calloc(1, sizeof(*s));
    if (!s)
        return NULL;
    s->config = *c;
    s->count = c->video_count + c->audio_count;
    size_t slots = c->kind == H3_SAMPLER_ER_SDE ? 6 :
        c->kind == H3_SAMPLER_RES_MULTISTEP ? 5 : 4;
    s->storage = calloc(slots * s->count, sizeof(float));
    if (masks)
        s->mask_storage = calloc(3 * s->count, sizeof(float));
    if (!s->storage || (masks && !s->mask_storage)) {
        free(s->storage);
        free(s->mask_storage);
        free(s);
        return NULL;
    }
    s->native = s->storage;
    s->carried = s->native + s->count;
    s->velocity = s->carried + s->count;
    s->prediction = s->velocity + s->count;
    if (c->kind == H3_SAMPLER_RES_MULTISTEP || c->kind == H3_SAMPLER_ER_SDE)
        s->old = s->prediction + s->count;
    if (c->kind == H3_SAMPLER_ER_SDE) s->old_derivative = s->old + s->count;
    if (masks) {
        s->anchor = s->mask_storage;
        s->mask = s->anchor + s->count;
        s->preserve_noise = s->mask + s->count;
    }
    return s;
}
void h3_sampler_state_free(h3_sampler_state *s) {
    if (s) {
        free(s->storage);
        free(s->mask_storage);
        free(s);
    }
}
const float *h3_sampler_state_video(const h3_sampler_state *s) {
    return s ? s->native : NULL;
}
const float *h3_sampler_state_audio(const h3_sampler_state *s) {
    return s ? s->native + s->config.video_count : NULL;
}
int h3_sampler_state_step(const h3_sampler_state *s) {
    return s ? s->step : -1;
}
const h3_sampler_config *h3_sampler_state_config(const h3_sampler_state *s) {
    return s ? &s->config : NULL;
}
static void pack(h3_sampler_state *s) {
    size_t nv = s->config.video_count;
    float c = s->config.video_shift / s->config.audio_shift,
          r = c + (1 - c) * s->config.schedule.video[s->step];
    memcpy(s->carried, s->native, nv * sizeof(float));
    for (size_t j = nv; j < s->count; j++)
        s->carried[j] = r * s->native[j];
}
h3_sampler_state *h3_sampler_state_start(const h3_sampler_config *c, const float *v, const float *a,
                                         const float *vc, const float *vm, const float *ac,
                                         const float *am, char *e, size_t z) {
    if (e && z)
        *e = 0;
    if (!valid_config(c) || !v || !a || (vm && !vc) || (am && !ac) ||
        !finite_values(v, c->video_count) || !finite_values(a, c->audio_count)) {
        reject(e, z, "invalid sampler config or native input");
        return NULL;
    }
    h3_sampler_state *s = allocate(c, vm || am);
    if (!s) {
        reject(e, z, "out of memory");
        return NULL;
    }
    s->step = c->start_step;
    s->rng = c->seed;
    memcpy(s->native, v, c->video_count * sizeof(float));
    memcpy(s->native + c->video_count, a, c->audio_count * sizeof(float));
    for (size_t j = 0; s->mask && j < s->count; j++) {
        size_t k = j < c->video_count ? j : j - c->video_count;
        const float *m = j < c->video_count ? vm : am, *clean = j < c->video_count ? vc : ac;
        float mask = m ? m[k] : 1, anchor = clean ? clean[k] : 0;
        if (!isfinite(mask) || mask < 0 || mask > 1 || !isfinite(anchor)) {
            h3_sampler_state_free(s);
            reject(e, z, "invalid clean latent or denoise mask");
            return NULL;
        }
        s->mask[j] = mask;
        s->anchor[j] = anchor;
        float sigma = j < c->video_count ? c->schedule.video[s->step] : c->schedule.audio[s->step];
        float noise = sigma > 0 ? (s->native[j] - (1 - sigma) * anchor) / sigma : 0;
        if (mask < 1 && !isfinite(noise)) {
            h3_sampler_state_free(s);
            reject(e, z, "preservation noise overflow");
            return NULL;
        }
        s->preserve_noise[j] = mask < 1 ? noise : 0;
        float preserve = sigma == 0 ? anchor : (1 - sigma) * anchor + sigma * s->preserve_noise[j];
        if (mask < 1)
            s->native[j] = mask == 0 ? preserve : mask * s->native[j] + (1 - mask) * preserve;
    }
    pack(s);
    if (!finite_values(s->carried, s->count)) {
        h3_sampler_state_free(s);
        reject(e, z, "native latent overflows packed coordinates");
        return NULL;
    }
    return s;
}
/* ComfyUI CONST/AV ER-SDE-Solver-3: lambda=sigma/(1-sigma), default
 * phi(lambda)=lambda*(exp(lambda^.3)+10), 200 left-endpoint quadrature
 * samples. Coefficients are computed once per transition, not per element. */
static float er_lambda(float sigma) { return sigma/(1-sigma); }
static float er_phi(float lambda) { return lambda*(expf(powf(lambda,.3f))+10); }
int h3_sampler_schedule_prepare(h3_sampler_kind kind,float vs,float as,
                                 h3_sigma_schedule *grid) {
    if (!grid || kind<H3_SAMPLER_EULER || kind>H3_SAMPLER_ER_SDE ||
        grid->steps<1 || grid->steps>H3_MAX_STEPS || !isfinite(vs) ||
        !isfinite(as) || vs<=0 || as<=0) return 0;
    if(kind==H3_SAMPLER_ER_SDE && grid->video[0]>=1) {
        /* percent_to_sigma receives a Python scalar upstream, then casts
         * to the sigma tensor's float32 dtype (not float32 shift algebra). */
        double base=1.0-1e-4;
        grid->video[0]=(float)(vs*base/(1+(vs-1)*base));
        grid->audio[0]=(float)(as*base/(1+(as-1)*base));
    }
    if(kind==H3_SAMPLER_ER_SDE &&
       (!isfinite(grid->video[0]) || !isfinite(grid->audio[0]) ||
        grid->video[0]>=1 || grid->audio[0]>=1))return 0;
    return 1;
}
static int er_step(h3_sampler_state *s,float *out) {
    const h3_sampler_config *c=&s->config;
    int i=s->step, age=i-c->start_step;
    float sigma=c->schedule.video[i],next=c->schedule.video[i+1];
    if(next==0) { memcpy(out,s->prediction,s->count*sizeof(float)); return 1; }
    float ls=er_lambda(sigma),lt=er_lambda(next);
    float ps=er_phi(ls),pt=er_phi(lt),r=pt/ps;
    float alpha=next/lt,a=alpha/(sigma/ls)*r,b=alpha*(1-r);
    float d1=0,d2=0,prev=0,span=0;
    if(age>=1) {
        float dt=lt-ls,step=-dt/200,sum=0,sum_u=0;
        for(int k=0;k<200;k++) {
            float pos=lt+(float)k*step, inv=1/er_phi(pos);
            sum+=inv; sum_u+=(pos-ls)*inv;
        }
        d1=alpha*(dt+sum*step*pt);
        prev=ls-er_lambda(c->schedule.video[i-1]);
        if(age>=2) {
            d2=alpha*(dt*dt/2+sum_u*step*pt);
            span=(ls-er_lambda(c->schedule.video[i-2]))/2;
        }
    }
    float noise=alpha*sqrtf(fmaxf(0,lt*lt-ls*ls*r*r));
    if(!isfinite(a)||!isfinite(b)||!isfinite(d1)||!isfinite(d2)||!isfinite(noise))return 0;
    for(size_t j=0;j<s->count;j++) {
        float derivative=age>=1?(s->prediction[j]-s->old[j])/prev:0;
        float second=age>=2?(derivative-s->old_derivative[j])/span:0;
        out[j]=a*s->carried[j]+b*s->prediction[j]+d1*derivative+
                       d2*second+noise*out[j];
    }
    return 1;
}
int h3_sampler_state_run(h3_sampler_state *s, int stop, h3_sampler_velocity evaluate, void *eo,
                         h3_sampler_noise noise, void *no, h3_sampler_observe observe, void *oo,
                         char *e, size_t z) {
    if (e && z)
        *e = 0;
    if (!s || !evaluate || stop < s->step || stop > s->config.schedule.steps)
        return reject(e, z, "invalid sampler run boundary");
    if (s->preview_only)
        return reject(e, z, "diagnostic preview state cannot resume sampling");
    const h3_sampler_config *c = &s->config;
    size_t nv = c->video_count, na = c->audio_count;
    float ratio = c->video_shift / c->audio_shift;
    if (noise && (c->kind == H3_SAMPLER_LCM || c->kind == H3_SAMPLER_ER_SDE))
        s->external_noise = 1;
    while (s->step < stop) {
        int step = s->step;
        float sigma = c->schedule.video[step], next = c->schedule.video[step + 1];
        uint64_t rng = s->rng;
        if (!evaluate(step, s->native, s->native + nv, s->velocity, s->velocity + nv, eo, e, z))
            return 0;
        if (!finite_values(s->velocity, s->count))
            return reject(e, z, "nonfinite model velocity");
        for (size_t j = 0; j < nv; j++)
            s->prediction[j] = s->native[j] + sigma * s->velocity[j];
        for (size_t j = nv; j < s->count; j++)
            s->prediction[j] = ratio * (s->native[j] + c->schedule.audio[step] * s->velocity[j]);
        if (!finite_values(s->prediction, s->count))
            return reject(e, z, "nonfinite clean prediction");
        /* Compute into velocity so failed callbacks leave the committed state intact. */
        /* RES history stores the raw model clean prediction, matching the
         * current prediction passed to the solver. Preservation is projected
         * exactly once onto the completed proposal below, never onto both the
         * prediction and proposal (which would square a soft mask). */
        if (c->kind == H3_SAMPLER_RES_MULTISTEP) {
            if (!h3_res_step(s->velocity, s->carried, s->prediction, s->history ? s->old : NULL,
                             s->count, c->schedule.video, step, c->schedule.steps))
                return reject(e, z, "RES solver rejected transition");
        } else if (c->kind == H3_SAMPLER_ER_SDE) {
            if(next!=0) {
                if(noise) {
                    if(!noise(step,0,s->velocity,nv,no,e,z) ||
                       !noise(step,1,s->velocity+nv,na,no,e,z))return 0;
                } else {
                    gaussian(s->velocity,nv,&rng);
                    gaussian(s->velocity+nv,na,&rng);
                }
                if(!finite_values(s->velocity,s->count))return reject(e,z,"nonfinite sampler noise");
            }
            if(!er_step(s,s->velocity))return reject(e,z,"ER-SDE solver rejected transition");
        } else if (c->kind == H3_SAMPLER_LCM) {
            if (next != 0) {
                if (noise) {
                    if (!noise(step, 0, s->velocity, nv, no, e, z) ||
                        !noise(step, 1, s->velocity + nv, na, no, e, z))
                        return 0;
                } else {
                    gaussian(s->velocity, nv, &rng);
                    gaussian(s->velocity + nv, na, &rng);
                }
                if (!finite_values(s->velocity, s->count))
                    return reject(e, z, "nonfinite sampler noise");
                for (size_t j = 0; j < s->count; j++)
                    s->velocity[j] = (1 - next) * s->prediction[j] + next * s->velocity[j];
            } else
                memcpy(s->velocity, s->prediction, s->count * sizeof(float));
            if (!finite_values(s->velocity, s->count))
                return reject(e, z, "nonfinite sampler transition");
        } else
            for (size_t j = 0; j < s->count; j++)
                s->velocity[j] =
                    next == 0 ? s->prediction[j]
                              : s->carried[j] +
                                    (next - sigma) * ((s->carried[j] - s->prediction[j]) / sigma);
        float r = ratio + (1 - ratio) * next;
        for (size_t j = 0; s->mask && j < s->count; j++) {
            float scale = j < nv ? 1 : r, m = s->mask[j];
            float sigma_next = j < nv ? next : c->schedule.audio[step + 1];
            float preserve = sigma_next == 0 ? s->anchor[j]
                                             : (1 - sigma_next) * s->anchor[j] +
                                                   sigma_next * s->preserve_noise[j];
            float native = s->velocity[j] / scale;
            if (m < 1) {
                native = m == 0 ? preserve : m * native + (1 - m) * preserve;
                s->velocity[j] = native * scale;
            }
        }
        if (!finite_values(s->velocity, s->count) || !finite_values(s->prediction, s->count))
            return reject(e, z, "nonfinite masked sampler transition");
        if(s->old_derivative && next!=0 && step>c->start_step) {
            float delta=er_lambda(sigma)-er_lambda(c->schedule.video[step-1]);
            /* Validate before committing either history or RNG. */
            for(size_t j=0;j<s->count;j++)
                if(!isfinite((s->prediction[j]-s->old[j])/delta))
                    return reject(e,z,"nonfinite ER-SDE derivative history");
            for(size_t j=0;j<s->count;j++)
                s->old_derivative[j]=(s->prediction[j]-s->old[j])/delta;
        }
        s->rng = rng;
        memcpy(s->carried, s->velocity, s->count * sizeof(float));
        if (s->old)
            memcpy(s->old, s->prediction, s->count * sizeof(float));
        for (size_t j = 0; j < s->count; j++)
            s->native[j] = s->carried[j] / (j < nv ? 1 : r);
        if (next == 0)
            for (size_t j = 0; s->mask && j < s->count; j++)
                if (s->mask[j] == 0)
                    s->native[j] = s->anchor[j];
        s->history = 1;
        s->step++;
        if (observe && !observe(s->step, s->native, s->native + nv, oo, e, z))
            return 0;
    }
    return 1;
}
/* Explicit payload: 12 uint64 words, digest, shifts, active sigma grid,
 * native/carried/clean history/anchors/masks/preservation noise; ER-SDE adds
 * derivative history and marks header word 10 with solver revision 1.
 * Older six-array files remain valid only for their original sampler kind. */
static size_t payload_size(const h3_sampler_config *c) {
    return 136 + 8 * (size_t)(c->schedule.steps + 1) +
        (c->kind == H3_SAMPLER_ER_SDE ? 28 : 24) * (c->video_count + c->audio_count);
}
static void config_words(const h3_sampler_config *c, uint64_t *h) {
    h[0] = (uint64_t)c->kind;
    h[1] = (uint64_t)c->schedule.steps;
    h[2] = (uint64_t)c->start_step;
    h[3] = c->video_count;
    h[4] = c->audio_count;
    h[5] = c->seed;
}
static int same_config(const h3_sampler_config *a, const h3_sampler_config *b) {
    return a->kind == b->kind && a->schedule.steps == b->schedule.steps &&
           a->start_step == b->start_step && a->video_count == b->video_count &&
           a->audio_count == b->audio_count && a->seed == b->seed &&
           !memcmp(&a->video_shift, &b->video_shift, 4) &&
           !memcmp(&a->audio_shift, &b->audio_shift, 4) &&
           !memcmp(a->request_digest, b->request_digest, 32) &&
           !memcmp(a->schedule.video, b->schedule.video, 4 * (a->schedule.steps + 1)) &&
           !memcmp(a->schedule.audio, b->schedule.audio, 4 * (a->schedule.steps + 1));
}
int h3_sampler_state_save(const h3_sampler_state *s, const char *path, char *e, size_t z) {
    if (!s || s->external_noise)
        return reject(e, z, "cannot persist sampler with external noise state");
    if (s->preview_only)
        return reject(e, z, "diagnostic preview state cannot be persisted");
    size_t n = payload_size(&s->config);
    unsigned char *b = calloc(1, n);
    if (!b)
        return reject(e, z, "out of memory");
    uint64_t h[12] = {0};
    config_words(&s->config, h);
    h[6] = (uint64_t)s->step;
    h[7] = (uint64_t)s->history;
    h[8] = s->rng;
    h[9] = s->mask != NULL;
    h[10] = s->config.kind == H3_SAMPLER_ER_SDE ? 1 : 0;
    memcpy(b, h, 96);
    memcpy(b + 96, s->config.request_digest, 32);
    memcpy(b + 128, &s->config.video_shift, 4);
    memcpy(b + 132, &s->config.audio_shift, 4);
    size_t o = 136, k = 4 * (size_t)(s->config.schedule.steps + 1);
    memcpy(b + o, s->config.schedule.video, k);
    o += k;
    memcpy(b + o, s->config.schedule.audio, k);
    o += k;
    const float *arrays[] = {s->native, s->carried, s->old, s->anchor, s->mask,
                            s->preserve_noise, s->old_derivative};
    for (int i = 0; i < (s->old_derivative ? 7 : 6); i++) {
        if (arrays[i])
            memcpy(b + o, arrays[i], s->count * 4);
        o += s->count * 4;
    }
    int ok = h3_native_write(path, 4, b, n, e, z);
    free(b);
    return ok;
}
static h3_sampler_state *load_state(const char *path, const h3_sampler_config *expected,
                                   char *e, size_t z) {
    void *raw = NULL;
    size_t n = 0;
    if (!h3_native_read(path, 4, &raw, &n, e, z))
        return NULL;
    unsigned char *b = raw;
    uint64_t h[12] = {0};
    h3_sampler_config c = {0};
    int ok = n >= 136;
    if (ok)
        memcpy(h, b, 96);
    ok = ok && h[0] <= H3_SAMPLER_ER_SDE && h[1] > 0 && h[1] <= H3_MAX_STEPS &&
         h[2] <= h[1] && h[3] <= SIZE_MAX && h[4] <= SIZE_MAX && h[6] >= h[2] && h[6] <= h[1] &&
         h[7] <= 1 && h[9] <= 1 && h[10] == (h[0] == H3_SAMPLER_ER_SDE ? 1u : 0u) && !h[11];
    if (ok) {
        c.kind = (h3_sampler_kind)h[0];
        c.schedule.steps = (int)h[1];
        c.start_step = (int)h[2];
        c.video_count = (size_t)h[3];
        c.audio_count = (size_t)h[4];
        c.seed = h[5];
        memcpy(c.request_digest, b + 96, 32);
        memcpy(&c.video_shift, b + 128, 4);
        memcpy(&c.audio_shift, b + 132, 4);
        ok = c.video_count <= SIZE_MAX - c.audio_count &&
             (c.video_count + c.audio_count) <= (SIZE_MAX - 136 - 8 * (h[1] + 1)) /
                 (c.kind == H3_SAMPLER_ER_SDE ? 28 : 24) &&
             n == payload_size(&c);
    }
    size_t o = 136, k = 4 * (size_t)(c.schedule.steps + 1);
    if (ok) {
        memcpy(c.schedule.video, b + o, k);
        o += k;
        memcpy(c.schedule.audio, b + o, k);
        o += k;
        ok = valid_config(&c) && (!expected || same_config(&c, expected)) && h[7] == (h[6] > h[2]);
    }
    h3_sampler_state *s = ok ? allocate(&c, (int)h[9]) : NULL;
    if (ok && !s)
        ok = 0;
    if (ok) {
        s->step = (int)h[6];
        s->history = (int)h[7];
        s->rng = h[8];
        float *arrays[] = {s->native, s->carried, s->old, s->anchor, s->mask,
                          s->preserve_noise, s->old_derivative};
        for (int i = 0; i < (s->old_derivative ? 7 : 6); i++) {
            if (arrays[i]) {
                memcpy(arrays[i], b + o, s->count * 4);
                if (!finite_values(arrays[i], s->count))
                    ok = 0;
            } else {
                for (size_t byte = 0; byte < s->count * 4; byte++)
                    if (b[o + byte])
                        ok = 0;
            }
            o += s->count * 4;
        }
        if(c.kind==H3_SAMPLER_ER_SDE) {
            int age=s->step-c.start_step;
            for(size_t j=0;j<s->count;j++)
                if((age==0 && s->old[j]!=0) || (age<2 && s->old_derivative[j]!=0))ok=0;
        }
        for (size_t j = 0; s->mask && j < s->count; j++)
            if (s->mask[j] < 0 || s->mask[j] > 1)
                ok = 0;
    }
    if (ok) {
        float ratio = c.video_shift / c.audio_shift,
              r = ratio + (1 - ratio) * c.schedule.video[s->step];
        for (size_t j = 0; j < s->count; j++) {
            float native = s->carried[j] / (j < c.video_count ? 1 : r);
            if (s->mask && s->mask[j] == 0 && s->step == c.schedule.steps)
                native = s->anchor[j];
            /* Initial native inputs need not be the exact round trip of their
             * packed audio coordinate. Both representations are persisted. */
            if (!isfinite(native) ||
                fabsf(native - s->native[j]) > 1e-6f * fmaxf(1.0f, fabsf(native)))
                ok = 0;
        }
    }
    free(raw);
    if (!ok) {
        h3_sampler_state_free(s);
        reject(e, z, "invalid or incompatible sampler state");
        return NULL;
    }
    return s;
}
h3_sampler_state *h3_sampler_state_load(const char *path, const h3_sampler_config *expected,
                                       char *e, size_t z) {
    if (!valid_config(expected)) {
        reject(e, z, "invalid expected sampler config");
        return NULL;
    }
    return load_state(path, expected, e, z);
}

h3_sampler_state *h3_sampler_state_load_preview(const char *path, char *e, size_t z) {
    h3_sampler_state *s = load_state(path, NULL, e, z);
    if (s)
        s->preview_only = 1;
    return s;
}

int h3_sampler_run(h3_sampler_kind kind, const h3_sigma_schedule *schedule, float vs, float as,
                   uint64_t seed, float *v, size_t nv, float *a, size_t na,
                   h3_sampler_velocity evaluate, void *eo, h3_sampler_noise noise, void *no,
                   h3_sampler_observe observe, void *oo, char *e, size_t z) {
    if (!schedule)
        return reject(e, z, "missing sampler schedule");
    h3_sampler_config c = {0};
    c.kind = kind;
    c.schedule = *schedule;
    c.video_shift = vs;
    c.audio_shift = as;
    c.seed = seed;
    c.video_count = nv;
    c.audio_count = na;
    h3_sampler_state *s = h3_sampler_state_start(&c, v, a, NULL, NULL, NULL, NULL, e, z);
    if (!s)
        return 0;
    int ok = h3_sampler_state_run(s, schedule->steps, evaluate, eo, noise, no, observe, oo, e, z);
    memcpy(v, s->native, nv * sizeof(float));
    memcpy(a, s->native + nv, na * sizeof(float));
    h3_sampler_state_free(s);
    return ok;
}
