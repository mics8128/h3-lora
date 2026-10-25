#ifndef H3_LATENT_H
#define H3_LATENT_H
#include <stddef.h>
#include <stdint.h>
typedef enum {
    H3_LATENT_TEMPORAL_NATIVE = 0,
    H3_LATENT_TEMPORAL_ENCODER_LINEAR_ENDPOINTS = 1
} h3_latent_temporal_mapping;
typedef struct {
    uint64_t video_frames, height, width, audio_frames;
    size_t video_count, audio_count;
    float *video, *audio;
    uint64_t source_frames;
    /* Native may omit source_frames (0). Linear endpoints requires it. When
     * present, source_frames must produce video_frames by generated cadence:
     * N<=5 ? 2 : ((N-5)/17)*5+2. Raw encoder cadence is not this schema. */
    uint32_t temporal_mapping;
} h3_latent;
/* Fixed native normalized schema: video_count=24*frames*height*width;
 * audio_count=64*audio_frames (32 channels x stereo). Spatial sizes are latent
 * sizes, not pixels. An absent stream has zero count and all its sizes zero.
 * At least one stream is required. load owns arrays; free releases them.
 * load output must be empty; save never takes ownership. */
int h3_latent_save(const char *, const h3_latent *, char *, size_t);
int h3_latent_load(const char *, h3_latent *, char *, size_t);
void h3_latent_free(h3_latent *);
/* Shared versioned native-file envelope. Payload is native-endian IEEE float;
 * incompatible endianness/float representation is rejected, never converted. */
int h3_native_write(const char *, uint32_t, const void *, size_t, char *, size_t);
int h3_native_read(const char *, uint32_t, void **, size_t *, char *, size_t);
#endif
