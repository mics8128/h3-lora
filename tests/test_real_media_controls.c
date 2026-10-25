#include "h3.h"
#include "h3_latent.h"
#include "h3_ffmpeg.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        exit(1); \
    } \
} while (0)

static void le16(unsigned char *p, uint16_t value) {
    p[0] = (unsigned char)value;
    p[1] = (unsigned char)(value >> 8);
}

static void le32(unsigned char *p, uint32_t value) {
    for (int i = 0; i < 4; i++) p[i] = (unsigned char)(value >> (8 * i));
}

static void write_three_second_wave(const char *path) {
    const uint32_t samples = 32000 * 3;
    unsigned char header[44] = {0};
    memcpy(header, "RIFF", 4);
    le32(header + 4, 36 + samples * 4);
    memcpy(header + 8, "WAVEfmt ", 8);
    le32(header + 16, 16);
    le16(header + 20, 1);
    le16(header + 22, 2);
    le32(header + 24, 32000);
    le32(header + 28, 128000);
    le16(header + 32, 4);
    le16(header + 34, 16);
    memcpy(header + 36, "data", 4);
    le32(header + 40, samples * 4);
    FILE *stream = fopen(path, "wb");
    CHECK(stream);
    CHECK(fwrite(header, 1, sizeof(header), stream) == sizeof(header));
    for (uint32_t i = 0; i < samples; i++) {
        int16_t amplitude = (int16_t)(2000 * sin((double)i * 440 * 6.283185307179586 / 32000));
        unsigned char frame[4];
        le16(frame, (uint16_t)amplitude);
        le16(frame + 2, (uint16_t)amplitude);
        CHECK(fwrite(frame, 1, sizeof(frame), stream) == sizeof(frame));
    }
    CHECK(fclose(stream) == 0);
}

typedef struct { int width, height, frames; } frame_expectation;

static int expect_frame(const h3_frame *frame, void *opaque) {
    frame_expectation *expected = opaque;
    CHECK(frame->width == expected->width);
    CHECK(frame->height == expected->height);
    CHECK(frame->stride == expected->width * 3);
    expected->frames++;
    return 0;
}

static void test_terminal_canvas(h3_ctx *ctx, const char *directory) {
    char encoded[256], sized[256], native[256], error[256];
    CHECK(snprintf(encoded, sizeof(encoded), "%s/av.latent", directory) > 0);
    CHECK(snprintf(sized, sizeof(sized), "%s/sized.mp4", directory) > 0);
    CHECK(snprintf(native, sizeof(native), "%s/native.mp4", directory) > 0);
    h3_latent latent = {.video_frames = 7, .height = 16, .width = 16,
        .audio_frames = 37, .video_count = 24 * 7 * 16 * 16,
        .audio_count = 64 * 37, .source_frames = 22};
    latent.video = calloc(latent.video_count, sizeof(float));
    latent.audio = calloc(latent.audio_count, sizeof(float));
    CHECK(latent.video && latent.audio);
    CHECK(h3_latent_save(encoded, &latent, error, sizeof(error)));
    h3_latent_free(&latent);
    h3_params params = H3_PARAMS_DEFAULT;
    params.width = params.height = 512;
    params.render_width = params.render_height = 256;
    params.frames = 22;
    params.steps = params.start_transition = 2;
    params.initial_latent_path = encoded;
    params.output_path = sized;
    frame_expectation expected = {512, 512, 0};
    params.on_frame = expect_frame;
    params.callback_opaque = &expected;
    h3_result *result = h3_generate(ctx, "Decode the existing terminal state.", &params);
    if (!result) fprintf(stderr, "%s\n", h3_last_error(ctx));
    CHECK(result);
    CHECK(result->width == 512 && result->height == 512 && result->frames == 22);
    CHECK(expected.frames == 22);
    h3_result_free(result);
    int width, height;
    CHECK(h3_ffprobe_visual_size(sized, &width, &height, error, sizeof(error)));
    CHECK(width == 512 && height == 512);
    expected = (frame_expectation){256, 256, 0};
    params.output_path = native;
    result = h3_decode_latent(ctx, encoded, 0, &params);
    if (!result) fprintf(stderr, "%s\n", h3_last_error(ctx));
    CHECK(result);
    CHECK(result->width == 256 && result->height == 256 && expected.frames == 22);
    h3_result_free(result);
    CHECK(h3_ffprobe_visual_size(native, &width, &height, error, sizeof(error)));
    CHECK(width == 256 && height == 256);
    CHECK(unlink(encoded) == 0);
    CHECK(unlink(sized) == 0);
    CHECK(unlink(native) == 0);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL_DIR\n", argv[0]);
        return 2;
    }
    char directory[] = "/tmp/h3-media-controls-XXXXXX";
    CHECK(mkdtemp(directory));
    char source[256], encoded[256], decoded[256];
    CHECK(snprintf(source, sizeof(source), "%s/source.wav", directory) > 0);
    CHECK(snprintf(encoded, sizeof(encoded), "%s/audio.latent", directory) > 0);
    CHECK(snprintf(decoded, sizeof(decoded), "%s/decoded.wav", directory) > 0);
    write_three_second_wave(source);
    h3_ctx *ctx = h3_load_dir(argv[1], NULL, 0, H3_LORA_FL2VA);
    if (!ctx) fprintf(stderr, "%s\n", h3_last_error(NULL));
    CHECK(ctx);
    h3_params params = H3_PARAMS_DEFAULT;
    params.frames = 22; /* Video duration must not truncate standalone audio. */
    params.latent_output_path = encoded;
    h3_result *result = h3_encode_media(ctx, source, 2, &params);
    if (!result) fprintf(stderr, "%s\n", h3_last_error(ctx));
    CHECK(result);
    h3_result_free(result);
    h3_latent latent = {0};
    char error[256];
    CHECK(h3_latent_load(encoded, &latent, error, sizeof(error)));
    CHECK(latent.video_count == 0);
    CHECK(latent.audio_frames == 120);
    CHECK(latent.audio_count == 32 * 2 * 120);
    h3_latent_free(&latent);
    params.output_path = decoded;
    params.latent_output_path = NULL;
    result = h3_decode_latent(ctx, encoded, 0, &params);
    if (!result) fprintf(stderr, "%s\n", h3_last_error(ctx));
    CHECK(result);
    CHECK(result->sample_rate == 32000);
    h3_result_free(result);
    struct stat status;
    CHECK(stat(decoded, &status) == 0);
    CHECK(status.st_size == 44 + 32000 * 3 * 2 * (off_t)sizeof(float));
    test_terminal_canvas(ctx, directory);
    h3_free(ctx);
    CHECK(unlink(source) == 0);
    CHECK(unlink(encoded) == 0);
    CHECK(unlink(decoded) == 0);
    CHECK(rmdir(directory) == 0);
    puts("real standalone audio duration and terminal output canvas tests passed");
    return 0;
}
