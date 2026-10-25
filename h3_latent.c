#define _POSIX_C_SOURCE 200809L
#include "h3_latent.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
static int bad(char *e, size_t n, const char *s) {
    if (e && n)
        snprintf(e, n, "%s", s);
    return 0;
}
static uint64_t hash(const unsigned char *p, size_t n) {
    uint64_t h = UINT64_C(14695981039346656037);
    while (n--)
        h = (h ^ *p++) * UINT64_C(1099511628211);
    return h;
}
int h3_native_write(const char *path, uint32_t type, const void *data, size_t n, char *e,
                    size_t z) {
    if (!path || !data || !n || sizeof(float) != 4 || FLT_RADIX != 2)
        return bad(e, z, "invalid native file arguments");
    unsigned char header[40] = {'H', '3', 'N', 'A', 'T', 'I', 'V', 'E'};
    uint32_t version = 1, endian = 0x01020304;
    uint64_t size = n, sum = hash(data, n);
    memcpy(header + 8, &version, 4);
    memcpy(header + 12, &type, 4);
    memcpy(header + 16, &endian, 4);
    memcpy(header + 24, &size, 8);
    memcpy(header + 32, &sum, 8);
    size_t len = strlen(path);
    if (len > SIZE_MAX - 12)
        return bad(e, z, "native path too long");
    char *tmp = malloc(len + 12);
    if (!tmp)
        return bad(e, z, "out of memory");
    snprintf(tmp, len + 12, "%s.XXXXXX", path);
    int fd = mkstemp(tmp);
    FILE *f = fd < 0 ? NULL : fdopen(fd, "wb");
    int ok = 0;
    if (f) {
        ok = fwrite(header, 1, 40, f) == 40 && fwrite(data, 1, n, f) == n && fflush(f) == 0 &&
             fsync(fd) == 0;
        if (fclose(f) != 0)
            ok = 0;
    } else if (fd >= 0)
        close(fd);
    if (ok)
        ok = rename(tmp, path) == 0;
    if (!ok)
        unlink(tmp);
    free(tmp);
    return ok ? 1 : bad(e, z, "cannot atomically write native file");
}
int h3_native_read(const char *path, uint32_t type, void **data, size_t *n, char *e, size_t z) {
    if (!path || !data || !n)
        return bad(e, z, "invalid native file arguments");
    *data = NULL;
    *n = 0;
    FILE *f = fopen(path, "rb");
    if (!f)
        return bad(e, z, "cannot open native file");
    unsigned char h[40];
    uint32_t v = 0, t = 0, endian = 0;
    uint64_t size = 0, sum = 0;
    struct stat st;
    int ok = fread(h, 1, 40, f) == 40;
    if (ok) {
        memcpy(&v, h + 8, 4);
        memcpy(&t, h + 12, 4);
        memcpy(&endian, h + 16, 4);
        memcpy(&size, h + 24, 8);
        memcpy(&sum, h + 32, 8);
    }
    ok = ok && !memcmp(h, "H3NATIVE", 8) && v == 1 && t == type && endian == 0x01020304 &&
         h[20] == 0 && h[21] == 0 && h[22] == 0 && h[23] == 0 && sizeof(float) == 4 &&
         FLT_RADIX == 2 && size > 0 && size <= SIZE_MAX - 40 && fstat(fileno(f), &st) == 0 &&
         st.st_size >= 40 && (uint64_t)st.st_size == size + 40;
    void *p = ok ? malloc((size_t)size) : NULL;
    ok = ok && p && fread(p, 1, (size_t)size, f) == size && fgetc(f) == EOF && !ferror(f);
    fclose(f);
    if (ok)
        ok = hash(p, (size_t)size) == sum;
    if (!ok) {
        free(p);
        return bad(e, z, "invalid, truncated or incompatible native file");
    }
    *data = p;
    *n = (size_t)size;
    return 1;
}
void h3_latent_free(h3_latent *l) {
    if (l) {
        free(l->video);
        free(l->audio);
        memset(l, 0, sizeof(*l));
    }
}
static int valid(const h3_latent *l) {
    if (!l || (!l->video_count && !l->audio_count) || l->video_count > SIZE_MAX - l->audio_count ||
        (l->video_count + l->audio_count) > (SIZE_MAX - 64) / 4 || l->temporal_mapping > 1 ||
        (l->temporal_mapping == 1 && (!l->source_frames || !l->video_count)))
        return 0;
    if (l->source_frames) {
        uint64_t expected_frames = l->source_frames <= 5 ? 2 :
            ((l->source_frames - 5) / 17) * 5 + 2;
        if (!l->video_count || l->video_frames != expected_frames)
            return 0;
    }
    if (l->video_count) {
        if (!l->video || !l->video_frames || !l->height || !l->width ||
            l->video_frames > SIZE_MAX / 24)
            return 0;
        size_t count = 24 * (size_t)l->video_frames;
        if (l->height > SIZE_MAX / count)
            return 0;
        count *= (size_t)l->height;
        if (l->width > SIZE_MAX / count || count * (size_t)l->width != l->video_count)
            return 0;
    } else if (l->video_frames || l->height || l->width)
        return 0;
    if (l->audio_count) {
        if (!l->audio || !l->audio_frames || l->audio_frames > SIZE_MAX / 64 ||
            64 * (size_t)l->audio_frames != l->audio_count)
            return 0;
    } else if (l->audio_frames)
        return 0;
    for (size_t i = 0; i < l->video_count; i++)
        if (!isfinite(l->video[i]))
            return 0;
    for (size_t i = 0; i < l->audio_count; i++)
        if (!isfinite(l->audio[i]))
            return 0;
    return 1;
}
int h3_latent_save(const char *p, const h3_latent *l, char *e, size_t z) {
    if (!valid(l))
        return bad(e, z, "invalid native latent shape or values");
    size_t n = 64 + 4 * (l->video_count + l->audio_count);
    unsigned char *b = calloc(1, n);
    if (!b)
        return bad(e, z, "out of memory");
    uint64_t h[7] = {l->video_frames, l->height,      l->width,        l->audio_frames,
                     l->video_count,  l->audio_count, l->source_frames};
    memcpy(b, h, 56);
    memcpy(b + 56, &l->temporal_mapping, 4);
    if (l->video_count)
        memcpy(b + 64, l->video, l->video_count * 4);
    if (l->audio_count)
        memcpy(b + 64 + l->video_count * 4, l->audio, l->audio_count * 4);
    int ok = h3_native_write(p, 5, b, n, e, z);
    free(b);
    return ok;
}
int h3_latent_load(const char *p, h3_latent *out, char *e, size_t z) {
    if (!out)
        return bad(e, z, "missing latent output");
    void *b = NULL;
    size_t n = 0;
    if (!h3_native_read(p, 5, &b, &n, e, z))
        return 0;
    uint64_t h[7] = {0};
    uint32_t mapping = 0, reserved = 0;
    if (n >= 64) {
        memcpy(h, b, 56);
        memcpy(&mapping, (char *)b + 56, 4);
        memcpy(&reserved, (char *)b + 60, 4);
    }
    h3_latent l = {h[0], h[1], h[2], h[3], 0, 0, NULL, NULL, h[6], mapping};
    int ok = n >= 64 && !reserved && h[4] <= SIZE_MAX && h[5] <= SIZE_MAX &&
             h[4] <= SIZE_MAX - h[5] && (h[4] + h[5]) <= (SIZE_MAX - 64) / 4 &&
             n == 64 + 4 * (h[4] + h[5]);
    if (ok) {
        l.video_count = (size_t)h[4];
        l.audio_count = (size_t)h[5];
        l.video = l.video_count ? malloc(l.video_count * 4) : NULL;
        l.audio = l.audio_count ? malloc(l.audio_count * 4) : NULL;
        ok = (!l.video_count || l.video) && (!l.audio_count || l.audio);
        if (ok) {
            if (l.video_count)
                memcpy(l.video, (char *)b + 64, l.video_count * 4);
            if (l.audio_count)
                memcpy(l.audio, (char *)b + 64 + l.video_count * 4, l.audio_count * 4);
            ok = valid(&l);
        }
    }
    free(b);
    if (!ok) {
        h3_latent_free(&l);
        return bad(e, z, "invalid native latent shape or values");
    }
    *out = l;
    return 1;
}
