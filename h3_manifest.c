/* Strict bounded execution manifest reader. All returned strings borrow document storage. */
#include "h3_cli.h"
#include "h3_host.h"
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static const char *published_path(const char *path) {
    struct stat info;
    return path && !stat(path, &info) && S_ISREG(info.st_mode) && info.st_size > 0 ? path : NULL;
}
typedef struct node {
    char *s;
    int type, next, child, used;
}
node;
typedef struct {
    char *text, *p;
    node n[8192];
    int count, failed;
}
document;
/* JSON input is UTF-8; reject overlong encodings, surrogates and truncation. */
static int valid_utf8(const unsigned char *s, size_t length) {
    for (size_t i=0; i<length;) {
        unsigned c=s[i++], code;
        int remaining;
        if (c<128) continue;
        if (c>=0xc2 && c<=0xdf) { code=c&31; remaining=1; }
        else if (c>=0xe0 && c<=0xef) { code=c&15; remaining=2; }
        else if (c>=0xf0 && c<=0xf4) { code=c&7; remaining=3; }
        else return 0;
        if ((size_t)remaining>length-i) return 0;
        int count=remaining;
        while (remaining--) {
            unsigned next=s[i++];
            if ((next&0xc0)!=0x80) return 0;
            code=(code<<6)|(next&63);
        }
        if ((count==2 && code<0x800) || (count==3 && code<0x10000) ||
            code>0x10ffff || (code>=0xd800 && code<=0xdfff)) return 0;
    }
    return 1;
}
static void ws(document *d) {
    while (isspace((unsigned char)*d->p)) d->p++;
}
static unsigned hex4(document *d) {
    unsigned v = 0;
    for (int i = 0; i < 4; i++) {
        unsigned char c = *d->p;
        if (!isxdigit(c)) {
            d->failed = 1;
            return 0;
        }
        d->p++;
        v = v * 16 + (c <= '9' ? c - '0' : tolower(c) - 'a' + 10);
    }
    return v;
}
static char *string(document *d) {
    if (*d->p++ != '"') {
        d->failed = 1;
        return NULL;
    }
    char *s = d->p, *out = s;
    while (*d->p && *d->p != '"') {
        unsigned char c = *d->p++;
        if (c < 32) {
            d->failed = 1;
            return s;
        }
        if (c == '\\') {
            c = *d->p++;
            switch (c) {
                case '"': case '\\': case '/': break;
                case 'b': c = '\b';
                break;
                case 'f': c = '\f';
                break;
                case 'n': c = '\n';
                break;
                case 'r': c = '\r';
                break;
                case 't': c = '\t';
                break;
                case 'u': {
                    unsigned u = hex4(d);
                    if (d->failed) return s;
                    if (u >= 0xd800 && u <= 0xdbff) {
                        if (d->p[0] != '\\' || d->p[1] != 'u') {
                            d->failed = 1;
                            return s;
                        }
                        d->p += 2;
                        unsigned low = hex4(d);
                        if (d->failed || low < 0xdc00 || low > 0xdfff) {
                            d->failed = 1;
                            return s;
                        }
                        u = 0x10000 + ((u - 0xd800) << 10) + low - 0xdc00;
                    }
                    else if (u >= 0xdc00 && u <= 0xdfff) {
                        d->failed = 1;
                        return s;
                    }
                    /* Public native strings cannot represent embedded NUL. */
                    if (!u) {
                        d->failed = 1;
                        return s;
                    }
                    if (u < 128) *out++ = (char)u;
                    else if (u < 2048) {
                        *out++ = (char)(192 | (u >> 6));
                        *out++ = (char)(128 | (u & 63));
                    }
                    else if (u < 65536) {
                        *out++ = (char)(224 | (u >> 12));
                        *out++ = (char)(128 | ((u >> 6) & 63));
                        *out++ = (char)(128 | (u & 63));
                    }
                    else {
                        *out++ = (char)(240 | (u >> 18));
                        *out++ = (char)(128 | ((u >> 12) & 63));
                        *out++ = (char)(128 | ((u >> 6) & 63));
                        *out++ = (char)(128 | (u & 63));
                    }
                    continue;
                }
                default: d->failed = 1;
                return s;
            }
        }
        *out++ = (char)c;
    }
    if (*d->p != '"') {
        d->failed = 1;
        return s;
    }
    d->p++;
    *out = 0;
    return s;
}
static int value(document *d, int depth) {
    ws(d);
    if (d->failed || depth > 24 || d->count >= 8191) {
        d->failed = 1;
        return 0;
    }
    int id = ++d->count;
    node *n = &d->n[id];
    if (*d->p == '"') {
        n->type = 's';
        n->s = string(d);
    }
    else if (*d->p == '{' || *d->p == '[') {
        int open = *d->p++;
        int close = open == '{' ? '}' : ']';
        n->type = open;
        ws(d);
        int *link = &n->child;
        while (*d->p != close && !d->failed) {
            if (open == '{') {
                ws(d);
                if (*d->p != '"') {
                    d->failed = 1;
                    break;
                }
                int key = value(d, depth + 1);
                *link = key;
                link = &d->n[key].next;
                ws(d);
                if (*d->p != ':') {
                    d->failed = 1;
                    break;
                }
                d->p++;
            }
            int child = value(d, depth + 1);
            *link = child;
            link = &d->n[child].next;
            ws(d);
            if (*d->p == close) break;
            if (*d->p != ',') {
                d->failed = 1;
                break;
            }
            d->p++;
            ws(d);
            if (*d->p == close) {
                d->failed = 1;
                break;
            }
        }
        if (*d->p != close) d->failed = 1;
        else d->p++;
    }
    else {
        n->type = 'n';
        char *end = d->p;
        while (*end && !isspace((unsigned char)*end) &&
        *end != ',' && *end != ']' && *end != '}') end++;
        size_t length = (size_t)(end - d->p);
        if (!length) {
            d->failed = 1;
            return id;
        }
        n->s = malloc(length + 1);
        if (!n->s) {
            d->failed = 1;
            return id;
        }
        memcpy(n->s, d->p, length);
        n->s[length] = 0;
        d->p = end;
    }
    return id;
}
static void release(document *d) {
    for (int i = 1; i <= d->count; i++)
    if (d->n[i].type == 'n') free(d->n[i].s);
    free(d->text);
    free(d);
}
static document *load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END)) {
        fclose(f);
        return NULL;
    }
    long size = ftell(f);
    if (size < 0 || size > 1048576 || fseek(f, 0, SEEK_SET)) {
        fclose(f);
        return NULL;
    }
    document *d = calloc(1, sizeof(*d));
    if (!d) {
        fclose(f);
        return NULL;
    }
    /* Extra zero bytes make bounded escape lookahead safe at EOF. */
    d->text = calloc((size_t)size + 8, 1);
    if (!d->text) {
        fclose(f);
        release(d);
        return NULL;
    }
    size_t got = fread(d->text, 1, (size_t)size, f);
    int bad = ferror(f);
    fclose(f);
    if (got != (size_t)size || bad || memchr(d->text, 0, got) ||
        !valid_utf8((const unsigned char *)d->text, got)) {
        release(d);
        return NULL;
    }
    d->p = d->text;
    value(d, 0);
    ws(d);
    if (d->failed || *d->p) {
        release(d);
        return NULL;
    }
    return d;
}
static int member(document *d, int obj, const char *name) {
    if (!obj || d->n[obj].type != '{') {
        d->failed = 1;
        return 0;
    }
    int found = 0;
    for (int key = d->n[obj].child; key;) {
        int v = d->n[key].next;
        if (!v) {
            d->failed = 1;
            return 0;
        }
        if (!strcmp(d->n[key].s, name)) {
            if (found) d->failed = 1;
            found = v;
            d->n[key].used = 1;
        }
        key = d->n[v].next;
    }
    return found;
}
static int known(document *d, int obj) {
    for (int key = d->n[obj].child; key;) {
        int v = d->n[key].next;
        if (!d->n[key].used) return 0;
        key = d->n[v].next;
    }
    return !d->failed;
}
static const char *text(document *d,int n) {
    if(!n||d->n[n].type!='s'||!d->n[n].s[0]) {
        d->failed=1;
        return NULL;
    }
    return d->n[n].s;
}
static int integer(document *d,int n,int min,int max) {
    if(!n||d->n[n].type!='n') {
        d->failed=1;
        return 0;
    }
    char *end;
    errno=0;
    long v=strtol(d->n[n].s,&end,10);
    if(errno||*end||v<min||v>max) {
        d->failed=1;
        return 0;
    }
    return (int)v;
}
static int json_number(const char *s) {
    const char *p=s;
    if(*p=='-')p++;
    if(*p=='0')p++;
    else {
        if(*p<'1'||*p>'9')return 0;
        while(isdigit((unsigned char)*p))p++;
    }
    if(*p=='.') {
        p++;
        if(!isdigit((unsigned char)*p))return 0;
        while(isdigit((unsigned char)*p))p++;
    }
    if(*p=='e'||*p=='E') {
        p++;
        if(*p=='+'||*p=='-')p++;
        if(!isdigit((unsigned char)*p))return 0;
        while(isdigit((unsigned char)*p))p++;
    }
    return !*p;
}
static void quote(const char *s) {
    if(!s) {
        fputs("null",stdout);
        return;
    }
    putchar('"');
    for(const unsigned char *p=(const unsigned char*)s;*p;p++) {
        if(*p=='"'||*p=='\\')printf("\\%c",*p);
        else if(*p<32)printf("\\u%04x",*p);
        else putchar(*p);
    }
    putchar('"');
}
static int fail(const char *code,const char *field,const char *message) {
    fputs("{\"schema_version\":1,\"event\":\"error\",\"error\":{\"code\":",stdout);
    quote(code);
    fputs(",\"field\":",stdout);
    quote(field);
    fputs(",\"message\":",stdout);
    quote(message);
    puts(",\"details\":{},\"repairs\":[],\"retryable\":false}}");
    fflush(stdout);
    return 2;
}
static int progress(const char *phase,int completed,int total,void *opaque) {
    (void)opaque;
    fputs("{\"schema_version\":1,\"event\":\"progress\",\"phase\":",stdout);
    quote(phase);
    printf(",\"completed\":%d,\"total\":%d}\n",completed,total);
    fflush(stdout);
    return 0;
}
typedef struct {
    const char *path;
    int bad;
}
control;
static int transition(int completed,int total,void *opaque) {
    control *c=opaque;
    progress("transition",completed,total,NULL);
    if(!c->path)return 0;
    FILE *f=fopen(c->path,"rb");
    if(!f) {
        if(errno!=ENOENT) {
            c->bad=1;
            return 2;
        }
        return 0;
    }
    fclose(f);
    document *d=load(c->path);
    if(!d) {
        c->bad=1;
        return 2;
    }
    const char *action=text(d,member(d,1,"action"));
    int result=0;
    if(action&&!strcmp(action,"pause"))result=1;
    else if(action&&!strcmp(action,"cancel"))result=2;
    else if(!action||strcmp(action,"run"))c->bad=1;
    if(!known(d,1)||d->failed)c->bad=1;
    release(d);
    return c->bad?2:result;
}
int h3_cli_capabilities(void) {
    const char *names[] = {
        "generate", "references", "image_guide", "clip_guide", "audio_guide", "clip_audio_guide", "initial_latent", "masks", "resume", "encode", "decode"
    }
    ;
    fputs("{\"schema_version\":1,\"native_version\":\"" H3_VERSION "\",\"event\":\"capabilities\",\"model_inventory_probed\":false,\"capabilities\":{", stdout);
    for (size_t i=0; i<sizeof(names)/sizeof(names[0]); i++) {
        if(i) putchar(',');
        quote(names[i]);
        fputs(":{\"mechanism\":",stdout);
        quote(names[i]);
        fputs(",\"availability\":\"available\",\"quality_evidence\":\"unverified\",\"requires_model\":true}",stdout);
    }
    puts("},\"samplers\":[\"euler\",\"lcm\",\"res_multistep\",\"er_sde\"],\"schedulers\":[\"serving\",\"simple\",\"beta\",\"beta57\"],\"constraints\":{\"default_execution\":\"ordinary_throughput\",\"pause_requires_control_path\":true,\"stateful_execution\":\"explicit_control_checkpoint_initial_mask_or_resume\",\"exact_resume_requires_uncached\":true,\"resume_requires_compatible_checkpoint\":true,\"control_schema\":{\"action\":[\"run\",\"pause\",\"cancel\"]}}}");
    return 0;
}
static int resolve_schedule(const h3_params *p, h3_sigma_schedule *schedule) {
    switch (p->scheduler) {
        case H3_SCHEDULER_SERVING:
            return h3_serving_schedule_build(p->steps, p->video_shift, p->audio_shift, schedule);
        case H3_SCHEDULER_SIMPLE:
            return h3_schedule_build(p->steps, p->video_shift, p->audio_shift, schedule);
        case H3_SCHEDULER_BETA: case H3_SCHEDULER_BETA57:
            return h3_beta_schedule_build(p->steps, p->video_shift, p->audio_shift,
                                         p->scheduler == H3_SCHEDULER_BETA57, schedule);
    }
    return 0;
}

/* Planning uses the same native schedule, including beta-grid deduplication. */
int h3_cli_schedule(const char *name, const char *steps,
                    const char *video_shift, const char *audio_shift) {
    h3_params p = H3_PARAMS_DEFAULT;
    if (!strcmp(name, "serving")) p.scheduler = H3_SCHEDULER_SERVING;
    else if (!strcmp(name, "simple")) p.scheduler = H3_SCHEDULER_SIMPLE;
    else if (!strcmp(name, "beta")) p.scheduler = H3_SCHEDULER_BETA;
    else if (!strcmp(name, "beta57")) p.scheduler = H3_SCHEDULER_BETA57;
    else return fail("invalid_request", "params.scheduler", "Unknown scheduler");
    char *end;
    errno = 0;
    long count = strtol(steps, &end, 10);
    if (errno || end == steps || *end || count < 1 || count > H3_MAX_STEPS)
        return fail("invalid_request", "params.steps", "Invalid requested transition count");
    p.steps = (int)count;
    const char *values[] = {video_shift, audio_shift};
    float *shifts[] = {&p.video_shift, &p.audio_shift};
    const char *fields[] = {"params.video_shift", "params.audio_shift"};
    for (size_t i = 0; i < 2; i++) {
        errno = 0;
        *shifts[i] = strtof(values[i], &end);
        if (errno || end == values[i] || *end || !isfinite(*shifts[i]) || *shifts[i] <= 0)
            return fail("invalid_request", fields[i], "Shift must be positive and finite");
    }
    h3_sigma_schedule schedule;
    if (!resolve_schedule(&p, &schedule))
        return fail("invalid_request", "params.steps", "Cannot construct the native schedule");
    printf("{\"schema_version\":1,\"requested_steps\":%d,\"transitions\":%d,\"video\":[",
           p.steps, schedule.steps);
    for (int i = 0; i <= schedule.steps; i++)
        printf("%s%.9g", i ? "," : "", (double)schedule.video[i]);
    fputs("],\"audio\":[", stdout);
    for (int i = 0; i <= schedule.steps; i++)
        printf("%s%.9g", i ? "," : "", (double)schedule.audio[i]);
    puts("]}");
    return 0;
}

#define INT_FIELD(name,min,max) do { \
    int field_node = member(d,po,#name); \
    if (field_node) p.name = integer(d,field_node,min,max); \
} while (0)
#define PATH_FIELD(name) do { \
    int field_node = member(d,po,#name); \
    if (field_node) p.name = text(d,field_node); \
} while (0)
#define BOOL_FIELD(name) do { \
    int field_node = member(d,po,#name); \
    if (field_node) { \
        const char *boolean = d->n[field_node].s; \
        if (d->n[field_node].type != 'n' || \
            (strcmp(boolean,"true") && strcmp(boolean,"false") && \
             strcmp(boolean,"0") && strcmp(boolean,"1"))) d->failed = 1; \
        else p.name = !strcmp(boolean,"true") || !strcmp(boolean,"1"); \
    } \
} while (0)
int h3_cli_manifest(const char *path) {
    document *d=load(path);
    if(!d)return fail("invalid_request","manifest","Unreadable, oversized or malformed JSON manifest");
    for(int i=1;i<=d->count;i++)if(d->n[i].type=='n' && strcmp(d->n[i].s,"true") && strcmp(d->n[i].s,"false") && strcmp(d->n[i].s,"null") && !json_number(d->n[i].s))d->failed=1;
    h3_params p=H3_PARAMS_DEFAULT;
    h3_reference refs[12]= {
        0
    }
    ;
    h3_guide guides[128]= {
        0
    }
    ;
    control ctl= {
        0
    }
    ;
    int version=integer(d,member(d,1,"schema_version"),1,1);
    (void)version;
    const char *op=text(d,member(d,1,"operation")),*model=text(d,member(d,1,"model_dir"));
    int prompt_node=member(d,1,"prompt");
    const char *prompt="";
    if(prompt_node) {
        if(d->n[prompt_node].type!='s')d->failed=1;
        else prompt=d->n[prompt_node].s;
    }
    const char *target=text(d,member(d,1,"target"));
    p.request_digest=text(d,member(d,1,"request_digest"));
    int n=member(d,1,"control_path");
    if(n)ctl.path=text(d,n);
    n=member(d,1,"checkpoint_path");
    if(n)p.checkpoint_path=text(d,n);
    int po=member(d,1,"params");
    if(!po||d->n[po].type!='{')d->failed=1;
    INT_FIELD(width,16,32768);
    INT_FIELD(height,16,32768);
    INT_FIELD(frames,1,INT_MAX);
    INT_FIELD(steps,1,1000);
    INT_FIELD(render_width,0,32768);
    INT_FIELD(render_height,0,32768);
    INT_FIELD(denoise_reuse,1,3);
    INT_FIELD(core_reuse,1,6);
    INT_FIELD(dit_layers,H3_MIN_DIT_LAYERS,H3_DEFAULT_DIT_LAYERS);
    INT_FIELD(start_transition,0,1000);
    BOOL_FIELD(token_reduction);
    BOOL_FIELD(use_int8_row_fc2);
    BOOL_FIELD(use_reference_rope);
    BOOL_FIELD(ssd_streaming);
    BOOL_FIELD(use_slower_bf16_mlp);
    BOOL_FIELD(use_slower_bf16_qkv);
    BOOL_FIELD(use_slower_bf16_attention_output);
    BOOL_FIELD(use_slower_row_major_attention_output);
    BOOL_FIELD(use_slower_unfused_int8_inputs);
    BOOL_FIELD(use_slower_unfused_qkv_rope);
    BOOL_FIELD(use_slower_scalar_qkv_rms);
    BOOL_FIELD(use_slower_uncached_int8_scales);
    BOOL_FIELD(use_slower_dynamic_fc1_k);
    BOOL_FIELD(use_slower_grouped_quantizer);
    BOOL_FIELD(preview_denoise);
    PATH_FIELD(output_path);
    PATH_FIELD(initial_latent_path);
    PATH_FIELD(mask_path);
    PATH_FIELD(resume_path);
    PATH_FIELD(latent_output_path);
    n=member(d,po,"checkpoint_path");
    if(n) {
        if(p.checkpoint_path)d->failed=1;
        p.checkpoint_path=text(d,n);
    }
    n=member(d,po,"seed");
    if(n) {
        char *end;
        errno=0;
        if(d->n[n].type!='n'||!isdigit((unsigned char)d->n[n].s[0]))d->failed=1;
        else {
            p.seed=strtoull(d->n[n].s,&end,10);
            if(errno||*end)d->failed=1;
        }
    }
    n=member(d,po,"sampler");
    if(n) {
        const char *s=text(d,n);
        if(s&&!strcmp(s,"euler"))p.sampler=H3_SAMPLER_EULER;
        else if(s&&!strcmp(s,"lcm"))p.sampler=H3_SAMPLER_LCM;
        else if(s&&!strcmp(s,"res_multistep"))p.sampler=H3_SAMPLER_RES_MULTISTEP;
        else if(s&&!strcmp(s,"er_sde"))p.sampler=H3_SAMPLER_ER_SDE;
        else d->failed=1;
    }
    n=member(d,po,"scheduler");
    if(n) {
        const char *s=text(d,n);
        if(s&&!strcmp(s,"serving"))p.scheduler=H3_SCHEDULER_SERVING;
        else if(s&&!strcmp(s,"simple"))p.scheduler=H3_SCHEDULER_SIMPLE;
        else if(s&&!strcmp(s,"beta"))p.scheduler=H3_SCHEDULER_BETA;
        else if(s&&!strcmp(s,"beta57"))p.scheduler=H3_SCHEDULER_BETA57;
        else d->failed=1;
    }
    n=member(d,po,"reference_image_size");
    if(n) {
        const char *s=text(d,n);
        if(s&&!strcmp(s,"match"))p.reference_image_size=H3_REFERENCE_IMAGE_MATCH;
        else if(s&&!strcmp(s,"max"))p.reference_image_size=H3_REFERENCE_IMAGE_MAX;
        else d->failed=1;
    }
    const char *shifts[]= {
        "video_shift","audio_shift"
    }
    ;
    for(int i=0;i<2;i++) {
        n=member(d,po,shifts[i]);
        if(n) {
            char *end;
            errno=0;
            float v=d->n[n].type=='n'?strtof(d->n[n].s,&end):0;
            if(d->n[n].type!='n'||errno||*end||!isfinite(v)||v<=0)d->failed=1;
            else if(i)p.audio_shift=v;
            else p.video_shift=v;
        }
    }
    const char *input=NULL;
    n=member(d,po,"input_path");
    if(n)input=text(d,n);
    int policy=3,diagnostic=0;
    n=member(d,po,"stream_policy");
    if(n)policy=integer(d,n,1,3);
    n=member(d,po,"diagnostic_preview");
    if(n)diagnostic=integer(d,n,0,1);
    const char *precision="f32";
    n=member(d,po,"vae_precision");
    if(n)precision=text(d,n);
    if(!precision || (strcmp(precision,"f32")&&strcmp(precision,"bf16")))d->failed=1;
    if(!known(d,po))d->failed=1;
    for(int which=0;which<2;which++) {
        int a=member(d,1,which?"guides":"references");
        if(!a)continue;
        if(d->n[a].type!='[') {
            d->failed=1;
            continue;
        }
        for(int item=d->n[a].child;item;item=d->n[item].next) {
            size_t count=which?p.guide_count:p.reference_count;
            if(count>=(which?64:12)) {
                d->failed=1;
                break;
            }
            const char *kind=text(d,member(d,item,"kind")),*source=text(d,member(d,item,"path"));
            const char *audio=NULL;
            int embedded=0;
            n=member(d,item,"audio_path");
            if(n)audio=text(d,n);
            n=member(d,item,"include_embedded_audio");
            if(n) {
                if(d->n[n].type!='n')d->failed=1;
                else if(!strcmp(d->n[n].s,"true"))embedded=1;
                else if(strcmp(d->n[n].s,"false"))d->failed=1;
            }
            if(which) {
                h3_guide *g=&guides[p.guide_count++];
                g->path=source;
                g->audio_path=audio;
                g->include_embedded_audio=embedded;
                g->frame_index=integer(d,member(d,item,"frame_index"),0,p.frames-1);
                n=member(d,item,"source_start_frame");
                if(n)g->source_start_frame=integer(d,n,0,INT_MAX);
                n=member(d,item,"source_frames");
                if(n)g->source_frames=integer(d,n,0,INT_MAX);
                if(kind&&!strcmp(kind,"image"))g->kind=H3_GUIDE_IMAGE;
                else if(kind&&!strcmp(kind,"clip"))g->kind=H3_GUIDE_CLIP;
                else if(kind&&!strcmp(kind,"audio"))g->kind=H3_GUIDE_AUDIO;
                else if(kind&&!strcmp(kind,"clip_audio"))g->kind=H3_GUIDE_CLIP_AUDIO;
                else d->failed=1;
            }
            else {
                h3_reference *r=&refs[p.reference_count++];
                r->path=source;
                r->audio_path=audio;
                r->include_embedded_audio=embedded;
                if(kind&&!strcmp(kind,"image"))r->kind=H3_REFERENCE_IMAGE;
                else if(kind&&!strcmp(kind,"video"))r->kind=H3_REFERENCE_VIDEO;
                else if(kind&&!strcmp(kind,"audio"))r->kind=H3_REFERENCE_AUDIO;
                else if(kind&&!strcmp(kind,"video_audio"))r->kind=H3_REFERENCE_VIDEO_AUDIO;
                else d->failed=1;
            }
            if(!known(d,item))d->failed=1;
        }
    }
    h3_lora_spec loras[64];
    size_t lora_count=0;
    int la=member(d,1,"loras");
    if(la) {
        if(d->n[la].type!='[')d->failed=1;
        else for(int item=d->n[la].child;item;item=d->n[item].next) {
            if(lora_count>=64) {
                d->failed=1;
                break;
            }
            const char *adapter=text(d,member(d,item,"path"));
            int strength=member(d,item,"strength");
            float scale=0;
            if(!strength||d->n[strength].type!='n')d->failed=1;
            else {
                char *end;
                errno=0;
                scale=strtof(d->n[strength].s,&end);
                if(errno||*end||!isfinite(scale))d->failed=1;
            }
            if(!known(d,item))d->failed=1;
            loras[lora_count++]=(h3_lora_spec) {
                adapter,scale
            }
            ;
        }
    }
    if(!known(d,1)||!op||!model||!prompt||!target||!p.request_digest)d->failed=1;
    if(p.request_digest) {
        if(strlen(p.request_digest)!=64)d->failed=1;
        else for(int i=0;i<64;i++)if(!isxdigit((unsigned char)p.request_digest[i]))d->failed=1;
    }
    if(target&&strcmp(target,"fl2va")&&strcmp(target,"ref2va"))d->failed=1;
    if(op&&strcmp(op,"generate")&&strcmp(op,"resume")&&strcmp(op,"encode")&&strcmp(op,"decode"))d->failed=1;
    if(op&&(!strcmp(op,"encode")||!strcmp(op,"decode"))&&!input)d->failed=1;
    if(op&&!strcmp(op,"resume")&&!p.resume_path)d->failed=1;
    if(ctl.path&&!p.checkpoint_path)d->failed=1;
    if (op && !strcmp(op,"encode") && !p.latent_output_path) d->failed=1;
    if (op && !strcmp(op,"decode") && !p.output_path) d->failed=1;
    if (op && (!strcmp(op,"generate") || !strcmp(op,"resume"))) {
        if (!p.output_path && !p.latent_output_path) d->failed=1;
        h3_sigma_schedule schedule;
        if (!resolve_schedule(&p, &schedule) || p.start_transition > schedule.steps) d->failed=1;
        else {
            int terminal_decode = p.initial_latent_path && p.start_transition == schedule.steps;
            if (!terminal_decode && target &&
                ((p.reference_count != 0) != !strcmp(target,"ref2va"))) d->failed=1;
        }
    }
    if (p.preview_denoise || p.start_transition>p.steps) d->failed=1;
    if (op && (!strcmp(op,"generate") || !strcmp(op,"resume")) &&
        (p.frames<5 || p.frames%17!=5)) d->failed=1;
    if(d->failed) {
        release(d);
        return fail("invalid_request","manifest","Unknown, duplicate, missing or invalid manifest field");
    }
    p.references=refs;
    p.guides=guides;
    p.on_progress=progress;
    p.on_transition=ctl.path ? transition : NULL;
    p.callback_opaque=&ctl;
    if(setenv("H3_VAE_PRECISION",precision,1)) {
        release(d);
        return fail("engine_failed","params.vae_precision","Cannot set explicit VAE precision");
    }
    h3_ctx *ctx=h3_load_dir(model,loras,lora_count,!strcmp(target,"ref2va")?H3_LORA_REF2VA:H3_LORA_FL2VA);
    if(!ctx) {
        int rc=fail(h3_last_error_code(NULL),h3_last_error_field(NULL),h3_last_error(NULL));
        release(d);
        return rc;
    }
    h3_result *r=!strcmp(op,"encode")?h3_encode_media(ctx,input,policy,&p):!strcmp(op,"decode")?h3_decode_latent(ctx,input,diagnostic,&p):!strcmp(op,"resume")?h3_resume(ctx,prompt,&p):h3_generate(ctx,prompt,&p);
    int rc=0;
    if(!r)rc=fail(h3_last_error_code(ctx),h3_last_error_field(ctx),h3_last_error(ctx));
    else if(ctl.bad)rc=fail("invalid_request","control_path","Malformed or unreadable control request");
    else if(r->status==1 && !published_path(p.checkpoint_path))
        rc=fail("output_failed","checkpoint_path","Paused execution did not publish a readable checkpoint");
    else if(!r->status && ((strcmp(op,"encode") && p.output_path && !published_path(p.output_path)) ||
                          (strcmp(op,"decode") && p.latent_output_path && !published_path(p.latent_output_path))))
        rc=fail("output_failed","outputs","Execution did not publish a requested regular output file");
    else {
        const char *event=r->status==1?"paused":r->status==2?"cancelled":"result";
        fputs("{\"schema_version\":1,\"event\":",stdout);
        quote(event);
        printf(",\"diagnostic_preview\":%s",r->diagnostic_preview?"true":"false");
        fputs(",\"outputs\":{\"media\":",stdout);
        quote(r->status || !strcmp(op,"encode") ? NULL : published_path(p.output_path));
        fputs(",\"latent\":",stdout);
        quote(r->status || !strcmp(op,"decode") ? NULL : published_path(p.latent_output_path));
        fputs(",\"checkpoint\":",stdout);
        quote(r->status ? published_path(p.checkpoint_path) : NULL);
        puts("}}");
        fflush(stdout);
    }
    h3_result_free(r);
    h3_free(ctx);
    release(d);
    return rc;
}
