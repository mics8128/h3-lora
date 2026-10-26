# Runtime and technical reference

Daily commands, current model choices and the documentation map are in
[README.md](README.md). This reference covers the complete runtime, C API,
samplers, optional approximations, diagnostics and validation contracts.

The serving/base examples and historical speed presets below are distinct
from the recommended Turbo/simple daily workflow. Do not combine their
reuse, reduced-layer or internal-canvas controls with Turbo/simple.
Source code is MIT; model and adapter licenses are separate.

Generic examples below assume a model snapshot at `./MiniMax-H3`; the current
`ted` deployment uses the external model roots configured in README.md. Do not
download another copy merely to match a relative example path.

## Immutable multi-LoRA loading

Repeat `--lora PATH STRENGTH` to compose independent signed, finite strengths
into the selected transformer weights at load time, before existing int8 quantization:

```sh
./h3 -d "$MODEL_DIR" \
  --lora "$LIGHTX_V12_LORA" 1.0 --lora "$UNLOCKED_V21_LORA" 0.5 \
  -p "A fully clothed adult in a blue jacket walks and waves in a sunny park." \
  --sampler res_multistep --scheduler simple --video-shift 12 --audio-shift 3 \
  --width 1344 --height 768 --frames 22 --steps 5 \
  --layers 50 --reuse 1 --core-reuse 1 -o outputs/park.mp4
```

This selects Lightx 768p v1.2 at strength 1 and Unlocked v2.1 at 0.5, the
Unlocked author's recommended starting strength. Lightx's card recommends
RES/simple with at least five steps; 12/3 shifts are adopted H3 defaults where
the card is silent. Unlocked instead recommends Euler ancestral: this example
deliberately prioritizes the Turbo sampler, not simultaneous compliance with
both cards. The aligned 1344x768 canvas and 22 frames give a 0.917-second smoke
clip, not a full quality comparison. See [the measured result](TURBO_DESIGN.md#combined-768p-smoke).
Both operands are mandatory; scientific notation and negative strengths work.
Zero strength disables an entry without opening its path (even a missing path).
Repeated files contribute repeatedly. Unknown/unsupported tensors fail rather
than being skipped. Active adapters require resident weights: SSD streaming is
rejected. `--lora-model fl2va` (default) targets FL2VA and permits first/last-frame
anchors; `--lora-model ref2va` validates against the actual Ref2VA transformer
and accepts ordered image, video, audio-only, and mixed references with `match`
or `max` image sizing. `match` avoids unnecessary image-token cost.
The immutable target must match the generation request. Without active adapters,
ordered references still automatically select Ref2VA as before.

The stack is immutable: `!status` reports it, and changing adapters/strengths
requires a new process/context. Cache clearing retains adapter snapshots;
source-file replacement does not change a live context. `--info` with active
adapters validates and snapshots their payloads, although base payloads remain
unloaded. Startup reports hashes, strengths, scaling/rank policy and backing bytes.

C callers use `h3_load_dir(model_dir, NULL, 0, H3_LORA_FL2VA)` without adapters,
or pass an array of `h3_lora_spec`, its count and an immutable `h3_lora_model`
target. Input paths/specifications may be freed after construction.
`h3_get_loras(ctx, &count)` borrows canonical active specifications until `h3_free`.
The appended `h3_params` fields select sampler, scheduler and video/audio shifts;
initialize with `H3_PARAMS_DEFAULT` to retain Euler/serving/12/3 behavior.

## Turbo sampler selection

`--sampler euler|lcm|res_multistep|er_sde --scheduler serving|simple|beta|beta57`
and `--video-shift FLOAT --audio-shift FLOAT` expose explicit numerical contracts.
The default Euler/serving/12/3 retains the existing GPU path and initialization.
Serving supports Euler only. Simple uses the discrete 1,000-entry Comfy schedule;
beta and beta57 use Beta(.6,.6) and Beta(.5,.7) quantiles on its999-index grid,
ties-to-even, adjacent deduplication and terminal zero. Effective transitions
can be fewer than requested. These explicit grids use packed audiovisual state,
not independent native audio/video clocks, and require50blocks, full canvas,
no token reduction and `--reuse 1 --core-reuse 1`.
LCM and ER-SDE use run-local seeded noise, not PyTorch-identical Gaussian draws.
ER-SDE retains raw clean prediction/derivative history and applies its first-sigma
offset before model timestep precomputation, preserving upstream initial-noising
ordering. Solver3/revision1 checkpoints add a seventh float array; old sampler
checkpoints retain their existing format. Unknown revisions are rejected by both
native and task asset inspection. See [beta5 setup and numerical provenance](EROS_BETA5_SETUP.md)
and [the earlier Astra Turbo specification](TURBO_DESIGN.md).
Sampler correctness and adapter loadability alone do not establish visual quality.
The [comparison runner and exact artifact inventory](COMPARISON.md) cover all 21
Turbo weights with per-family settings and a separately labelled DMAD sampler
discrepancy. [Upstream PR/fork research](UPSTREAM_RESEARCH.md) keeps performance
patches separate from the model comparison and records hardware-specific evidence.

See [the Astra design and evidence record](LORA_DESIGN.md) for supported formats,
layout provenance, exact artifact hashes and measured acceptance results.
The two recorded artifacts passed individual and combined benign generation
in BF16 and M5 default modes; this is not a claim of general style fidelity or
four-step Turbo quality. Code remains MIT; model licenses are separate.
Never commit weights, merged weights or media.

## Tutorial

For this workflow, use the local clone only for source/docs and Git management.
Run all builds, tests, adapter loading and inference on the capable remote
machine in a separate checkout; keep real weights and generated media outside
the checkout. The commands below run there, not on the resource-limited local
machine. Local fixtures must remain tiny and synthetic.

### 1. Build and inspect the model

The examples assume that the Hugging Face snapshot is in `./MiniMax-H3` and
that FFmpeg and FFprobe are available on `PATH`.

```sh
make -j8
mkdir -p outputs
./h3 --info -d ./MiniMax-H3
```

`--info` checks the model layout and prints the selected Metal device without
mapping all weights or generating media. Run `./h3 --help` for the complete CLI
reference.

Without `-p`, the same binary starts an Iris-style interactive session:

```sh
./h3 -d ./MiniMax-H3 --width 512 --height 512 --steps 6
```

Type a prompt to generate a numbered video. The session keeps prompt conditioning,
the resident DiT weights and video decoder in memory. Changing only the seed reuses
the prepared request; changing the prompt, steps, canvas or references rebuilds the
affected request state without reloading compatible full-core weights. An unchanged
sigma schedule and conditioning-modality policy also reuse AdaLN precomputation.
Useful commands are `!status`, `!seed random`, `!seconds 2`, `!show`, `!save output.mp4`,
and `!cache`. Use `!help` for the full, short list.

This is automatic in the existing interactive mode: no new cache or backend flags.
`!cache clear` explicitly releases the retained model and decoder. Reference/anchor
edits no longer force a full cache clear; owned input signatures select the correct
conditioning and request state. SSD streaming, token reduction, thinned cores and
small-sequence precision-policy transitions retain conservative reload behavior.
Changing adapter weights/strengths still requires a new context. A one-shot `-p`
process exits after its output, so it cannot retain RAM for a later process.

Cold loading also composes large LoRA targets using two bounded, independent row
workers, preserving the original 128-row/64-rank arithmetic. Each block's enabled
weight quantizers share one synchronous submission; numerical precision, steps,
resolution and layers are unchanged. See [Astra's proposal](ASTRA_OPTIMIZATION_PLAN.md)
and [the research / measured results](COMFY_CACHE_RESEARCH.md).

First/last-frame conditioning is persistent in the session:

```text
h3> !first opening.png
h3> !last ending.png
h3> The camera moves slowly around the subject.
```

Use `!first clear` or `!last clear` to remove an anchor. Generated videos are
written to the session directory printed at startup.

For a general Ref2VA conditioning image, use `!ref-image PATH` instead. Images
are appended in order and exposed to the model as `<Picture 1>`, `<Picture 2>`,
and so on; filenames have no meaning to the model.

```text
h3> !ref-image person.png
h3> Make the person shown in Picture 1 wave to the camera.
```

`!refs` lists the current order, `!ref-remove N` removes one entry, and
`!refs clear` removes them all. Ordered Ref2VA references may coexist with
`!first`/`!last` and timed guides; the selected LoRA target must still match Ref2VA.

### 2. Make a first fast video

Start with the validated balanced preset. It generates 22 frames at 24 fps
(about 0.92 seconds), displays the evolving middle-video frame after every
denoising transition in a supported graphical terminal, and prints phase
timings:

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind." \
  --width 512 --height 512 \
  --frames 22 --steps 20 \
  --layers 45 --reuse 2 \
  --show \
  -o outputs/fox-fast.mp4
```

This is deliberately not the most aggressive configuration:

- `--steps 20` performs the default 20 denoising passes.
- `--reuse 2` computes 11 fresh denoiser velocities instead of all 20 and
  extrapolates the skipped transitions.
- `--layers 45` runs 45 of the 50 transformer blocks, reducing both time and
  unified-memory use.
- `--show` is optional. It supports Kitty/Ghostty and
  iTerm2/WezTerm/Konsole graphical protocols. It loads a resident preview VAE,
  displays one representative middle-video frame after every Euler transition,
  and then displays all final frames. Display dimensions default to 2x so the
  image has its intended logical size on macOS Retina screens; use `--zoom 1`
  on a non-HiDPI display. This adds preview decode time and roughly 10 GiB of
  temporary model residency; runs without `--show` are unchanged.
- `--profile` is optional and does not select a different generation path.

The first process invocation also pays model loading and filesystem-cache
costs. Compare performance using repeated runs, and alternate variants when
the machines are warming up because this workload is sensitive to thermal
throttling.

For a very short iteration, request four denoising passes directly:

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur." \
  --width 512 --height 512 --frames 22 \
  --steps 4 --layers 50 --reuse 1 \
  --show \
  -o outputs/fox-four-step.mp4
```

`--steps N` always means exactly N denoising passes. Four through seven passes
use the same schedule that won the low-budget comparison; increasing from 4
to 7 progressively improves detail and motion. Keep `--reuse 1` at such small
budgets so every requested pass runs the model. `--show` displays one preview
after each pass.

Several tail-heavy schedules were evaluated because most visible cleanup
happens late in a long run. They preserved too few early composition updates
and produced woven texture, weak motion, or clipped colors. The retained mode
uses the released linear base grid with one terminal point. On the 512-square,
22-frame fox test, the selected four-pass result had 0.556 full-video SSIM
against a 29-pass reference; an independent surfer test measured 0.547. The
four-pass denoise took about 3.5 seconds on M5 Max, versus 26.4 seconds for the
reference.

For a low-memory run, add `--ssd-streaming`:

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 50 --reuse 1 --ssd-streaming \
  -o outputs/fox-ssd.mp4
```

This uses the original BF16 checkpoint without conversion or quantization. It
keeps two DiT blocks in memory and reads the next block from SSD while the GPU
runs the current one. On M5 Max, tracked DiT storage fell from about 36.5 GiB to
2.0 GiB at 512 square and 2.1 GiB at 864x480. A warm 50-block forward measured
1.35 versus 2.49 seconds at 512 square (84% slower), and 2.14 versus 2.68
seconds at 864x480 (26% slower). These are comparisons against the same
full-residency BF16 path, and the results were byte-identical in both checks.

The 2.0--2.1 GiB figure is the DiT's tracked tensor storage, not total system
RAM. Prompt encoding and the two VAEs run in separate phases rather than adding
their full peaks to it; the OS, media buffers, and output resolution still need
headroom. `--show` keeps a preview VAE resident and adds roughly 10 GiB, so omit
it for the lowest-memory run.

SSD streaming is an explicit memory/speed tradeoff and is not the default. It
cannot be combined with `--use-int8-row-fc2`. In an interactive session, use
`!ssd-streaming on`.

### 3. Move toward reference quality

Change one control at a time when evaluating quality. First restore all layers,
then all denoiser evaluations, and finally raise the default 20-pass schedule
to the slower 50-pass reference:

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind." \
  --width 512 --height 512 \
  --frames 22 --steps 50 \
  --layers 50 --reuse 1 \
  -o outputs/fox-close.mp4
```

The defaults are `--steps 20 --layers 50 --reuse 1`; keep `--steps 50`
explicit for this close path. It performs 50 complete 50-block denoiser
forwards and is much more expensive than the default, but is the right oracle
when a fast mode changes the subject, anatomy, motion, or composition.
Numerical pixel identity with MLX is not expected because the random-number and
execution engines differ; the depicted content and motion should agree.

### 4. Choose a speed/quality preset

These are historical serving/base examples, not the current CLI defaults or
Turbo/simple presets. Raw CLI defaults are 20 steps, 50 blocks, reuse/core-reuse
1, no token reduction and no smaller internal canvas. Use README.md for Turbo.

| Control | Slow reference | Historical balanced example | Aggressive | Main impact |
|---|---:|---:|---:|---|
| Denoising passes | `--steps 50` | `--steps 20` | `--steps 4..7` | The number always names actual denoising passes. |
| Whole denoiser reuse | `--reuse 1` | `--reuse 2` | `--reuse 3` | At 20 steps: 20, 11, or 8 fresh DiT evaluations. |
| Active DiT blocks | `--layers 50` | `--layers 45` | `--layers 40` | Fewer blocks reduce compute and resident transformer weights. |
| Core residual reuse | `--core-reuse 1` | `--core-reuse 4` | `--core-reuse 6` | Refreshes patch/head work every step but runs the expensive core less often. |
| Token reduction | off | optional | `--token-reduction` | Pairs horizontal video tokens inside middle blocks; faster but may change composition. |
| Internal canvas | output size | `384x384` for 512 square output | `320x320` | Runs DiT/VAE smaller, then upscales with vImage. |

On M5, `--use-int8-row-fc2` uses one activation scale per FC2 row and a single
full-width TensorOps product. It is optional because it is less numerically
conservative than grouped int8. It reduced complete denoiser forwards by about
2.6% in reciprocal tests. Matched four-step fox and surfer videos kept the same
subjects, setting, and motion (full-video SSIM 0.919 and 0.828). In the
interactive session, use `!int8-row-fc2 on`.

`--reuse` and `--core-reuse` are mutually exclusive. Layer thinning can be
combined with either one.

To make the first command faster while keeping its output resolution, add
token reduction:

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A surfer riding inside a sharp blue ocean wave, one rider and one white board, realistic spray." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 45 --reuse 2 --token-reduction \
  -o outputs/surfer-fast.mp4
```

At the validated 512 square shape, token reduction cut the `45 layers + reuse
2` denoise profile from 16.69 to 12.60 seconds on the IT M5 Max. Independent
fox and surfer renders stayed coherent, but composition can diverge more from
the close path.

For an aggressive preview, render internally at 320 square and upscale to the
requested 512 square output:

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A red fox walking through snow, realistic, tracking shot." \
  --width 512 --height 512 \
  --render-width 320 --render-height 320 \
  --frames 22 --steps 20 --layers 40 --reuse 3 \
  -o outputs/fox-aggressive.mp4
```

This combination produced a clean, recognizable 22-frame fox in validation,
but loses fine detail and can change framing. Do **not** add `--token-reduction`
to both `--layers 40` and `--reuse 3`: that tested combination produced color
ringing, outlines, and ghosted limbs.

As an alternative to whole-velocity reuse, this keeps the timestep-dependent
patch and output heads fresh at every transition:

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A surfer riding a blue ocean wave." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 45 --core-reuse 4 \
  -o outputs/surfer-core-reuse.mp4
```

Use `--core-reuse 6` only as an aggressive preview. Values above 6 are not
exposed because validation lost subject fidelity.

### 5. Pick resolution and duration

Width and height must each be multiples of 32, at least 32, and their product
must not exceed `768 * 1344` pixels. Those are mechanical limits, not a promise
that every tiny canvas has good model quality. H3-Base is a 768p model.

| Canvas | Current guidance |
|---|---|
| `512x512` | Safest development size; repeatedly validated with multiple prompts. |
| `768x768` | Validated close-quality square output; substantially more expensive. |
| `1344x768`, `768x1344` | Released 768p-class landscape/portrait limit. |
| `1024x768`, `768x1024` | Valid 4:3 and 3:4 768p-class canvases. |
| `384x384` internal to `512x512` | Validated fast-quality scaling point. |
| `320x320` internal to `512x512` | Validated aggressive scaling point. |
| `256x256` | Native fast-preview canvas with automatic low-resolution RoPE adaptation. |

For a fast native 256-square preview:

```sh
./h3 -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest." \
  --width 256 --height 256 \
  --frames 22 --steps 20 \
  --layers 50 --reuse 1 \
  -o outputs/fox-256.mp4
```

At 256 square, H3 has only an `8x8` effective spatial-token grid, so it has less
room for fine detail and complex composition. H3 automatically halves spatial
RoPE coordinates at exactly 256 square. This removed repeating lattice
artifacts in long fox renders and stayed coherent on an independent portrait,
without adding tokens or runtime. Use `--use-reference-rope` to restore the
released/MLX coordinates for parity checks. Keep token reduction off at this
size. Native 128 square remains unsupported: its `4x4` token grid did not
recover a recognizable subject even with adjusted RoPE.

`--render-width` and `--render-height` must be set together, must have the same
aspect ratio as the output, and cannot exceed the output dimensions. The model
and VAE use the internal size; terminal frames and the encoded video retain the
requested output size.

H3 emits 24 fps and aligns frame requests upward to `5 + 17*n`:

Use `--seconds N` for a duration-oriented request, or `--frames N` for direct
frame control; the two options are mutually exclusive. Fractional seconds are
accepted. Seconds are converted at 24 fps and then rounded upward to the next
legal H3 temporal shape, so `--seconds 10` produces 243 frames (10.125 seconds).

| Frames | Approximate video duration |
|---:|---:|
| 22 | 0.917 seconds |
| 39 | 1.625 seconds |
| 56 | 2.333 seconds |
| 107 | 4.458 seconds |
| 243 | 10.125 seconds |
| 362 | 15.083 seconds |

Short clips are useful for development. The released workflow is intended for
roughly 4–15 second videos. A request such as `--frames 23` is rounded up to 39
frames rather than producing an arbitrary temporal shape.

### 6. Improve the prompt

A short prompt works, but the released system expects a Context-IR-like
description. State the subject, action, setting, camera, lighting/style, and
desired sound. For example:

```text
Scene: a single red fox in a snow-covered pine forest at dawn.
Action: the fox walks steadily left to right and looks toward the camera once.
Camera: medium-height lateral tracking shot, 50 mm lens, stable framing.
Look: photorealistic fur, cold blue ambient light, warm sunrise rim light.
Audio: soft footsteps in snow, light wind through pine branches, no music.
```

Keep identity and object counts explicit when they matter. `--seed N` controls
the native random stream; the default is 42. Compare options with the same
prompt, seed, resolution, frame count, and step count.

### 7. Preview frames and diagnose performance

- `--show` displays a representative frame after every denoising transition,
  followed by all frames from the completed video. Like Iris, it advertises 2x
  display dimensions by default for Retina terminals; `--zoom N` changes that
  factor without resizing the generated video or the encoded terminal image.
- `--frames-dir DIR` writes final callback frames as PPM files. Intermediate
  `--show` previews are not written there.
- `-o ''` disables MP4 encoding; combine it with `--frames-dir` when FFmpeg is
  unavailable.
- `--profile` reports phase wall time, Metal encoding/wait time, peak live
  tensor storage, cumulative allocation, and dispatch counts.

For example:

```sh
./h3 --profile -d ./MiniMax-H3 -p "A hummingbird hovering over red flowers." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 45 --reuse 2 --frames-dir outputs/hummingbird-frames \
  -o ''
```

### 8. Add image, video, and audio references

First/last-frame anchors alone use FL2VA; when combined with ordered references they become fixed timed guides on the Ref2VA path. Neither mechanism guarantees pixel-identical output frames:

```sh
./h3 -d ./MiniMax-H3 -p "The fox keeps walking through the snow." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 45 --reuse 2 \
  --first-frame fox.png --last-frame fox-later.png \
  -o outputs/fox-anchored.mp4
```

Ordered references select the distinct Ref2VA checkpoint. Use the flag matching
the media semantics:

```sh
# One image reference.
./h3 -d ./MiniMax-H3 -p "Use the animal and setting in the reference." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --ref-image fox.png -o outputs/fox-reference.mp4

# Continue a clip but ignore its soundtrack.
./h3 -d ./MiniMax-H3 -p "Continue the motion in this clip." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --ref-silent-video fox.mp4 -o outputs/fox-video-reference.mp4

# Preserve the clip's embedded audio.
./h3 -d ./MiniMax-H3 -p "Continue this audiovisual scene." \
  --width 512 --height 512 --frames 56 --steps 20 \
  --ref-video fox-with-audio.mp4 -o outputs/fox-video-audio.mp4

# Replace a video's soundtrack explicitly.
./h3 -d ./MiniMax-H3 -p "Continue the scene with the supplied music." \
  --width 512 --height 512 --frames 56 --steps 20 \
  --ref-video-audio silent-fox.mp4 replacement.wav \
  -o outputs/fox-replaced-audio.mp4

# An ordered image plus standalone audio reference.
./h3 -d ./MiniMax-H3 -p "Use the animal and music from the references." \
  --width 512 --height 512 --frames 56 --steps 20 \
  --ref-image fox.png --ref-audio music.wav \
  -o outputs/fox-image-audio.mp4
```

Reference flags may be repeated and their command-line order is preserved.
Standalone audio may be used alone or with image/video references. Audio references
must be 2–15 seconds; at most three audio inputs are accepted and their total
decoded duration is capped at 15 seconds. Embedded/replacement video soundtracks
also require at least two seconds of decoded video: request at least 56 output
frames and supply a source video of at least two seconds.

### Ref2VA with LoRA: one loader, distinct model targets

All reference kinds use the same `--lora PATH STRENGTH` loader; there is no separate
image/video/audio LoRA file-format selector. Select a **Ref2VA-trained** adapter and
`--lora-model ref2va` for these requests. FL2VA Turbo weights are not interchangeable
with Ref2VA Turbo, even when tensor shapes happen to validate.

The loader recognizes native/Comfy H3 fused-QKV tensor names and explicit Kohya
aliases in safetensors, with F32/BF16/F16 factors and supported alpha declarations.
It does **not** accept arbitrary Diffusers/PEFT split-QKV modules or `.default`
factor suffixes. Prefer the published **ComfyUI** artifact; do not rename keys or
assume that the `.safetensors` extension establishes compatibility. Unknown keys
fail instead of being silently ignored.

```sh
# MODEL_DIR contains the actual Ref2VA checkpoint plus required shared assets.
# REF_TURBO points to the Ref2VA v1.0 8-step ComfyUI weight, not FL2VA v1.2.
./h3 -d "$MODEL_DIR" --lora "$REF_TURBO" 1 --lora-model ref2va \
  --scheduler simple --sampler res_multistep --steps 8 \
  --layers 50 --reuse 1 --core-reuse 1 --video-shift 12 --audio-shift 3 \
  --width 1344 --height 768 --frames 56 \
  --ref-image person.png --ref-image product.png \
  -p 'The adult from <Picture 1> holds the product from <Picture 2>.' \
  -o outputs/person-product.mp4
```

Replace the two image flags with the required reference input:

| Intent | Flags |
|---|---|
| Motion/camera reference, ignoring source sound | `--ref-silent-video motion.mp4` |
| Audio-only reference | `--ref-audio ambience.wav` |
| Image plus audio | `--ref-image person.png --ref-audio ambience.wav` |
| Video with its embedded soundtrack | `--ref-video source-with-sound.mp4` |
| Video with an explicit replacement soundtrack | `--ref-video-audio motion.mp4 replacement.wav` |
| Separate video/audio references | `--ref-silent-video motion.mp4 --ref-audio ambience.wav` |

Use `<Picture N>`, `<Video N>`, and `<Audio N>` in the prompt; numbering is per
modality, following reference order. Paired soundtrack replacement is conditioning,
not a guarantee that output audio is copied bit-exactly. Reference motion/camera
guidance likewise does not guarantee exact trajectory transfer.

Pure-audio presentation follows the [ComfyUI code path](https://github.com/Comfy-Org/ComfyUI/blob/7cebca52227c63b4180ceed448cbd7ad22903b6d/comfy/text_encoders/minimax.py):
ordinary `<Audio N>: ` text plus prompt, AudioVAE latent conditioning, no fake
image, no chat template, and no waveform input to Qwen. ModelTC's Diffusers wrapper
has a stricter visual-reference requirement; these are different entry contracts.
See [related-project research and actual wiring results](REF_COMPATIBILITY_RESEARCH.md).

## Tests and runtime requirements

```sh
make test
make parity
```

`make test` runs the deterministic host suite and, when the ignored MLX fixture
is installed under `misc/fixtures/`, compiles the Metal source at runtime and
checks a complete toy H3 block against named MLX outputs. Runtime compilation is
intentional: it follows Iris and does not require Xcode's optional offline Metal
toolchain. The test covers both an F32 diagnosis path and the production BF16
storage path; wide BF16 matrix products and SDPA use cached MPSGraph graphs, with
direct Metal correctness fallbacks. `make parity` runs only those Metal/MLX
checks.

Fixture-free `h3_reduction_tests` is included in `make test`: independent softmax
and GroupNorm/SiLU oracles exercise multi-SIMD reductions, token boundaries
32/33/168/257, and repeated identical GPU inputs. Shared reduction scratch must
not be overwritten before every SIMD group has consumed the previous phase's
maximum or mean. These checks caught and now guard the Qwen causal-GQA and visual
encoder GroupNorm synchronization races; this is a correctness fix, not a lossy
speed mode.

Fixture-free `h3_vae_precision_tests` is also included in `make test`. Literal
BF16 inputs and small F32 biases verify widening-before-bias, row broadcast
and preservation of F32 output precision; dtype, capacity and dispatch-size
boundaries reject invalid requests. It does not require downloaded fixtures.

Real-model cache regression is opt-in:

```sh
make h3_real_cache_test
./h3_real_cache_test "$MODEL_DIR" "$TURBO_LORA" "$NEW_OUTPUT_DIR"
# Use the actual Ref2VA root and a matching Ref2VA LoRA for multimodal checks:
./h3_real_cache_test "$REF_MODEL_DIR" "$REF_TURBO_LORA" "$NEW_REF_OUTPUT_DIR" person.png ambience.wav
# Use the literal 'none' in place of the adapter path to check no-LoRA residency.
```

It renders warm transitions and compares native RGB plus fully decoded audio
against unique fresh-context oracles. Coverage includes seed, different-length
prompt, steps/back, shape/back, explicit cache clear, cancellation/recovery, and
optional audio removal/restoration. Restored settings reuse an already verified
oracle instead of rendering duplicate fresh inputs. Five-step sampling here is
an equivalence smoke, not a quality recommendation for every Turbo weight.

FFmpeg and FFprobe must be available on `PATH` for media inputs and MP4 output
(`H3_FFMPEG` and `H3_FFPROBE` may select explicit executables). Generated RGB24 and
32 kHz stereo F32 PCM are fed through concurrent pipes; no intermediate
uncompressed media file is created.

## Implementation and performance notes

The remainder documents the implementation behind the tutorial presets and the
environment variables retained for exact A/B diagnosis.

### Sampler and DiT controls

The default sampler uses the released shifted video/audio schedule. `--steps`
always names the number of denoising passes, with terminal zero added after the
last pass. Whole-denoiser reuse evaluates the first and last pass plus every
requested interval, then extrapolates skipped video and audio velocities on
their independent schedules. With very small step counts, keep `--reuse 1`.

For the low-budget path, the released linear base grid won against
actual-video-sigma linear spacing,
quadratic and cubic warps, exact 30-point tail subsets, mild power warps,
zero-order held full-grid velocities, linear velocity extrapolation, and RES.
The more tail-heavy candidates often sharpened the subject but damaged motion
or left a repetitive woven background; sparse RES and long extrapolation
intervals failed much more visibly.

Layer thinning ranks the checkpoint's actual AdaLN gates while protecting
structurally important first and final blocks. Unused weights and schedule
tensors are not retained, so `--layers 45` and `--layers 40` reduce both
transformer time and unified-memory use. Core reuse holds the previous full
transformer residual while refreshing the patch projection and timestep-aware
head; it remains mutually exclusive with whole-velocity reuse.

### Exact DiT fusions

Every active DiT block fuses its attention residual gate with the following MLP
AdaLN. The rounded BF16 residual is still written exactly, but the same row is
kept in threadgroup memory for normalization, eliminating one dispatch and one
global reread. Away from token-reduction boundaries, the MLP residual gate also
produces the next block's attention AdaLN and carries that normalized state
across the loop. `H3_DISABLE_FUSED_GATE_ADALN=1` and
`H3_DISABLE_FUSED_CROSS_BLOCK_ADALN=1` restore the two-kernel oracles.
The final audio/video AdaLN kernels bind directly to offsets in the residual
stream, avoiding two slice blits and 18.8 MiB of scratch at 512x512 (29.4 MiB
at the 864-class benchmark shape).
`H3_DISABLE_FUSED_FINAL_SLICE=1` restores the copy-plus-AdaLN oracle at load.
The BF16 final heads then apply AdaLN while loading their 16x16 projection
tiles, preserving the standalone rounding and accumulation order while
removing another equally sized normalized activation. The two optimizations
together save 37.5/58.9 MiB. `H3_DISABLE_FUSED_FINAL_HEAD=1` restores the
offset-AdaLN-plus-linear oracle at load.

### Token-reduction internals

`--token-reduction` is an independent aggressive DiT mode. After block 3 it
pairs adjacent horizontal target-video tokens while leaving text, audio,
conditions, and reference tokens exact. The complete full-resolution state is
kept as a bypass. During the first ten noisy evaluations it restores before
block 40; subsequent detail-forming evaluations restore before block 30. Each
token returns as its original value plus the update learned by its pair, so
within-pair detail is not discarded.
The pooling kernel writes only true-pair baselines into a dense tail of the
already allocated attention scratch buffer; odd-width singleton tokens need no
baseline. The full bypass uses the oversized QKV tail when it fits, with a
guarded dedicated fallback only for reference-heavy layouts. Common text-only
canvases therefore add no activation arena at any token-grid width. Pooling
also snapshots both source tokens while their BF16 values are already in
registers, avoiding a separate full-hidden blit and redundant source read. The
same entry kernel keeps each pooled row in threadgroup memory and emits the
first reduced block's attention AdaLN, eliminating another global residual read.
At the restore boundary, the first full-resolution attention AdaLN is fused
into expansion: a 10.5 KiB threadgroup row avoids a global residual reread while
still writing the exact bypass needed by the following residual branch.
On a thermal-balanced 512x512x22, 19-forward IT M5 Max A/B this reduced denoise
time from 39.13 to 28.06 seconds (28.3%). Final video/audio latent relative L2
was 5.56%/15.14%. First/middle/last fox frames retained one clean muzzle,
coherent legs, and sharp fur; an independent surfer remained consistent with
one rider and board through the wave spray. It changes composition and is
therefore opt-in rather than the close-reference default.
`H3_TOKEN_REDUCTION_BLOCKS` can override the later `4:30` interval;
`H3_TOKEN_REDUCTION_EARLY=STEPS:END` overrides the early schedule and `0`
disables it. `H3_DISABLE_TOKEN_REDUCTION=1` provides an in-context exact oracle.
`H3_DISABLE_FUSED_TOKEN_POOL_ADALN=1` and
`H3_DISABLE_FUSED_TOKEN_ADALN=1` independently restore the two-kernel entry and
exit boundaries for diagnosis.
Token reduction composes cleanly with the validated `--layers 45 --reuse 2`
settings: on the same 512 benchmark it reduced that profile from 16.69 to
12.60 seconds (24.5% marginal), and independent fox and surfer renders stayed
coherent. Do not combine it with both `--layers 40` and `--reuse 3`; that
6.47-second experiment produced chromatic ringing and ghosted limbs despite
acceptable latent norms.

### Internal canvas and video VAE

`--render-width` and `--render-height` run the model and VAE on a lower
same-aspect internal canvas, then high-quality vImage-scale RGB frames to the
requested output size before callbacks, terminal display, and encoding. This is an
explicit quality/speed tradeoff: a measured 384-to-512 prompt render reduced
M5 DiT time by 33% and video-VAE time by 18% while retaining a clean,
recognizable photorealistic result. Both values must be multiples of 32; the
exact output canvas remains the default.
For square 512 output, 384 is the fast-quality point and 320 is the validated
aggressive point. The latter produced a coherent walking fox and repeated at
8.02 seconds of DiT versus about 15.82 seconds natively. Native 256 uses the
same-cost spatial-RoPE adaptation described above; it remains a fast composition
preview rather than a substitute for a 512- or 768-class final render.
The video VAE automatically chooses a 256-320 pixel spatial tile from the
requested canvas geometry, minimizing repeated overlap work while keeping peak
storage bounded. `H3_VAE_TILE_PIXELS=256` restores the original conservative
tile plan for close-reference diagnosis.

The native video decoder defaults to F32. To opt into BF16 large projections
and BF16 attention without changing the model, sampler, steps or canvas:

```sh
H3_VAE_PRECISION=bf16 ./h3 -d ./MiniMax-H3 -p "A red fox walks in snow." -o outputs/fox.mp4
```

Only `f32` and `bf16` are accepted; unset means `f32`, while empty or other
values fail explicitly. Norms, RoPE, SwiGLU, residuals, small projections and
the audio VAE remain F32. Projection results round to BF16 before widening
and adding F32 bias, so this is **not bit-exact** with F32. Resident decoder
identity includes precision; changing it does not invalidate compatible DiT
weights. No additional weights, cache builder, daemon, int8 VAE or background
prefetch are required. LightVAE is a separate pruned-model proposal, **not**
implemented by this precision option; see [the assessment](COMFY_CACHE_RESEARCH.md#lightvae-是否值得做成選項).

Direct Metal pipelines are now built on first operation use and retained per
GPU context, following [upstream PR75](https://github.com/antirez/h3.c/pull/75).
Shader-library compilation remains eager. An unused kernel no longer blocks
context creation; a required kernel that cannot compile fails its operation.
This does not change arithmetic or make a GPU context thread-safe. BF16 VAE
arithmetic follows [PR73](https://github.com/antirez/h3.c/pull/73), without its
disk cache, prefetch or int8 modes.

### Weight residency and streamed prompt encoding

On M5-class GPUs, persistent transformer weights are mapped directly from their
safetensor shards instead of copied into anonymous shared buffers. This keeps
the 37 GiB model file-backed/reclaimable and slightly improves total transformer
time; M3 uses the faster copied-buffer path. `H3_ZERO_COPY_WEIGHTS=0` disables
the M5 selection for diagnostics.
The streamed Qwen text encoder preallocates a small ring of future layer
buffers and fills them on eight I/O workers while Metal executes the current
layer. The default ring depth is two layers on M3/older hardware and three on
M5, where the target machine has 128 GiB. `H3_QWEN_PREFETCH=0` restores the
single-layer synchronous reference path; values 1-8 select the worker count,
and `H3_QWEN_PREFETCH_DEPTH=1` through `6` overrides the ring depth.

`--ssd-streaming` is a separate, more aggressive residency mode for the DiT.
Only its small per-block normalization weights remain resident. Two complete
BF16 matrix slots alternate while a background reader fills the next slot in
checkpoint-offset order; the current Metal command buffer runs concurrently.
Darwin uncached reads avoid retaining a second copy in the filesystem cache.
The first active block is prefetched again during the final block, so a cached
interactive DiT is ready for its next denoiser evaluation. Measurements reached
about 13--14.6 GiB/s from the internal SSD. `H3_PROFILE=1` reports total bytes,
read throughput, and the part of the read wait that was not hidden by GPU work.

### Metal 4 and TensorOps paths

M5 GPUs automatically use native BF16 Metal 4/TensorOps for the DiT QKV and
attention-output projections at sequence lengths up to 2,048. The compact
Morton schedule routes Q/K/V directly into head-major attention inputs, avoids
three MPSGraph input transposes, and is byte-identical to the portable path. It
improves a complete 512x512 50-block forward by about 2% across repeated IT/US
M5 Max runs. For 2,049-3,072 rows, including 864x480, two row-offset Morton
dispatches preserve the efficient tile geometry and improve the complete
forward by about 2% in balanced runs. Still larger sequences stay on MPSGraph.
`H3_NAX=0` disables TensorOps for exact A/B diagnosis. The selection is guarded
at runtime and falls back to the unchanged portable library if compilation is
unavailable.

`H3_NAX=1` forces the broader native BF16 linear path. It passes the complete
50-block MLX fixture, but remains opt-in: exact-shape microbenchmarks favor its
128-row tile while full DiT runs currently favor MPSGraph scheduling. This
keeps a working NAX integration available for later quantized/fused kernels
without making a benchmark regression the default.
`H3_NAX=mlp` selects a more specialized Metal 4 path: paired FC1 gate/up
TensorOps tiles apply SwiGLU in threadgroup memory and write only the
14,336-wide activated intermediate, then FC2 also stays on TensorOps.
`H3_DISABLE_NAX_MLP=1` keeps the MPSGraph MLP in a context created this way for
same-process A/B testing. The path is deliberately opt-in because scheduling
depends on the OS GPU stack: the primary macOS 26.5.2 M5 Max gained 1.3-2.0%
in isolated real-weight MLP runs but lost about 1-3% in a complete 50-block forward,
while an otherwise identical macOS 26.5 M5 Max gained 1.4% in a same-context
forward A/B. The resulting 50-block velocities were close (1.9% video and 2.4%
audio relative L2), but not byte-identical.

### Specialized projection kernels

The narrow DiT audio/video output heads convert their small released F32
weights to BF16 once and use the Iris-derived 16x16 tiled linear directly on
BF16 activations. At the production 320-render geometry, isolated paired-head
measurements are 2.30x faster on M3 Max and 1.83x faster on M5 Max, with
relative L2 `8.64e-4`; the absolute M5 saving is about 0.6 ms per evaluated
step. Full fox and surfer sequences remained clean and measured 29.9/38.4 dB
against the F32-head renders. `H3_DIT_F32_FINAL=1` restores the close-reference
head and its extra activation buffers.
The F32 `96->5376` video and `32->5376` audio patch projections use a dedicated
16x16 cooperative tile, retaining F32 weights, inputs and accumulation while
rounding the tile result directly to BF16.
Paired production-shape measurements are 1.77x faster on M3 and 1.62-1.78x
on M5; the complete generated RGB stream is byte-identical to the scalar path.
Fusing the final cast improves the 2835-row tile itself from 2.499 to 1.734 ms
on M3 and 1.555 to 1.186 ms on M5, and removes 38.27/59.66 MiB of F32 scratch
at 512/864-class geometry. `H3_DISABLE_FUSED_PATCH_CAST=1` restores the tiled
F32 output plus standalone cast; `H3_SCALAR_PATCH=1` selects the scalar
diagnostic path.
The same tile binds its output directly into the packed hidden stream, removing
the BF16 media staging buffers and their blits. This saves another 19.13/29.83
MiB and improves the 2835-row boundary from 1.847 to 1.730 ms on M3 and 1.282
to 1.184 ms on M5. Contiguous T2VA uses byte offsets; FL2VA/Ref2VA use compact
destination-row maps so each modality remains one large dispatch. A complete
six-segment Ref2VA M5 ABBA remained byte-identical and improved 5.067 to 5.033
seconds per measured forward pair. `H3_DISABLE_FUSED_PATCH_PACK=1` restores the
staging buffers and packing blits.

### Scheduling and activation memory

The DiT core is split into two ordered Metal command buffers so GPU execution
of the first part overlaps CPU encoding of the second. Thermal-balanced ABBA
measurements select a 60%-depth split on M5 (30/50, 27/45, and 24/40), with
roughly 0.5-1.8% wins; M3 automatically splits only the validated 30/50 case,
which measured 1.2% faster, because 24/40 regressed there. The operation order
and generated bytes are unchanged. `H3_DIT_COMMAND_BLOCKS=0` restores one
command buffer; values 1-50 override the split for further tuning.
DiT activation buffers also follow their actual intra-block lifetimes: the QKV
projection arena is reused first for attention heads and then for the normalized
MLP input, while the current attention-output arena becomes the MLP output after
its branch has been consumed. This removes 61.25 MiB at 512-class geometry and
99.63 MiB at 864-class geometry without changing dispatches or arithmetic.
`H3_DISABLE_DIT_ACTIVATION_ALIAS=1` restores separate diagnostic buffers.
MPSGraph tensor-data wrappers for immutable DiT weights and biases are retained
with their resident buffers. This avoids rebuilding the same binding metadata
for every block and denoiser evaluation without copying tensor storage; measured
ABBA gains were 1.6% on M3 Max and 0.4-1.1% on M5 Max. Activation wrappers stay
transient because retaining them regressed the M5. The outputs remain
byte-identical, and `H3_DISABLE_GRAPH_DATA_CACHE=1` restores transient wrappers
for all tensors.
On M3/older hardware, the four MPSGraph segments in each DiT block also reuse
one `MPSCommandBuffer` wrapper for their shared underlying Metal command buffer.
Repeated thermal-balanced runs measured 1.0-1.6% faster on M3 Max; M5 measured
neutral, so it retains fresh wrappers. `H3_REUSE_MPS_COMMAND=0` or `1` overrides
the automatic selection. Results are byte-identical.
On M5, the serving Euler sampler keeps its patch-packed F32 latents and cached
BF16 velocities in Metal buffers. Each selected denoiser refresh is completed
before the next is encoded, avoiding MPSGraph back-pressure while removing all
intermediate latent/velocity readbacks and repacking. Two warm eight-run A/B
sequences measured small 0.1% and 0.3% gains with byte-identical final latents;
the path also saves roughly 16 bytes of transient host state per video-latent
element (about 136 MB at the 768p shape). M3 and older GPUs retain the CPU
sampler by default. `H3_CPU_SAMPLER=1` restores it on M5;
`H3_GPU_SAMPLER=1` selects the GPU-state path explicitly, and
`H3_GPU_SAMPLER_WINDOW=0` enables the slower unbounded encode-ahead diagnostic.

### Checkpoint layout and media pipeline

The released checkpoint stores DiT QKV rows interleaved per attention head.
Native Metal consumes that layout directly in the fused QK-normalization/RoPE
kernel, avoiding a checkpoint transpose and extra RAM. The earlier identity
interpretation was the cause of the noisy diagnostic outputs.

The public generation path decodes the joint audio latent with a streamed native
BigVGAN/AudioVAE and writes synchronized H.264 plus 32 kHz stereo AAC. The native
waveform agrees with the corrected MLX oracle to relative L2 `6.94e-5`.
`--first-frame`, `--last-frame`, and their combination use the released visual
VAE encoder, Qwen3-VL vision tower and three-deepstack multimodal presentation,
0.999 condition augmentation, and fixed condition rows in the native DiT. The
first image is stretched to the target canvas; the last image is aspect-cover
scaled and center cropped, matching the reference implementation. `--ref-image`
selects the distinct Ref2VA transformer, preserves ordered `<Picture N>`
presentation, and uses the released down-only aspect-preserving reference canvas.
`--ref-silent-video` additionally performs bounded 24 fps decoding, the visual
VAE's causal `ceil(T/4)` compression, two-frame Qwen sampling, and timestamped
`<Video N>` presentation. `--ref-video` preserves an embedded soundtrack,
`--ref-video-audio VIDEO AUDIO` supplies an explicit replacement, and
`--ref-audio` appends an ordered standalone clip. Reference audio is decoded as
32 kHz stereo F32 and encoded by the native AudioVAE posterior-mean path.
Reference audio uses clean augmentation strength **1.0**, pinned to the audio
condition timestep 1.0, and packed as width-32 rows on the rotary timeline of visual
references. Audio inputs are 2-15 seconds, at most three are
accepted, and their total decoded duration is capped at 15 seconds. Standalone
audio-only Ref2VA is supported, including active Ref2VA LoRAs; its audio latent
uses the target canvas timeline without a fabricated visual reference.

The native audio encoder matches the corrected MLX oracle at relative L2
`3.59e-6` on a real two-second stereo fixture. The correction is important: the
original MLX reshape interleaved left/right samples, whereas the official
PyTorch/SGLang path folds intact stereo channels into the batch dimension. On
the 128 GB M5 Max, clean end-to-end image+audio and embedded-video+audio renders
completed in 74.58 and 76.99 seconds respectively, each with about a 40.1 GB
peak physical footprint and zero swaps.

### Profiling and diagnostic paths

`--profile` reports each Metal-backed phase separately: wall time, CPU-side
command encoding, complete commit-to-fence wait, root-command GPU timestamps,
peak live tensor storage, cumulative allocation, and dispatch counts. The wait
measurement is the complete command turnaround; the root GPU timestamp alone
can omit child buffers scheduled internally by MPSGraph and is labeled
accordingly.

The DiT fast path evaluates each BF16 `fc1 -> SwiGLU -> fc2` block as one cached
graph, avoiding separate graph boundaries and persistent intermediate tensors.
Set `H3_DISABLE_FUSED_MLP=1` to retain the close-reference operation boundaries
for numerical diagnosis.

On supported M5 Metal 4 TensorOps hardware, the native int8 MLP engine is the
default. It dynamically quantizes activations, uses per-output-channel weight
scales, and gives the sensitive FC2 input one scale per 1,024 channels.
The selected FC2 kernel keeps scaled partial products in private cooperative
fragments instead of repeatedly spilling a 32 KiB threadgroup tile. A fixed
50-layer, 19-transition 512x512 render measured 36.30 seconds with BF16 MPS and
25.80 seconds with int8 on M5 Max. Beginning, middle, and final decoded frames
retained the same subject, composition, and motion; small edge and fur details
can differ. The current diagnostic implementation retains both BF16 and int8
MLP weights only when an A/B diagnostic requests them. Normal int8 loading
releases each block's BF16 FC1/FC2 buffers after their submitted quantization
finishes, reducing measured peak tensor storage to 25.9 GiB from the BF16
path's 36.4 GiB. Runtime weight quantization still adds startup time.

The fastest M5 path also quantizes each DiT QKV projection and writes its
Q/K/V tiles directly in head-major attention layout before the existing Q/K
normalization and RoPE kernel. In a fixed 50-layer, 19-transition 512x512
render this reduced denoising again, from 25.80 to 19.32 seconds. Sampled
beginning, middle, and final frames remained a coherent detailed fox walking
through snow; quantized attention can change framing and fine detail. Use
`--use-slower-bf16-qkv` for the close-reference BF16 projection. Normal int8
loading releases the redundant BF16 QKV weights after quantization.

The following attention-output projection is int8 as well on the default M5
path. Crossed same-model tests improve a complete forward by another 4.5-5.5%
at 512 and 864. A decoded fox render remained clean and closely matched the
int8-QKV-only composition; its thermally hot denoise measured 19.18 seconds.
Use `--use-slower-bf16-attention-output` to retain that projection in BF16.

On that int8 path, SDPA now leaves its result in native
`[head,row,dimension]` order. A specialized 256-thread kernel gathers and
quantizes each H3 row directly into the projection's row-major int8 buffer,
eliminating the intervening full-width BF16 transpose without changing any
output byte. Thermally controlled crossed runs improve complete 512 and 864
forwards by roughly 0.2-1.2%. Use
`--use-slower-row-major-attention-output` to restore the explicit BF16
row-major SDPA output and ordinary quantizer.

The M5 path also folds QKV and MLP activation quantization into the preceding
gated AdaLN kernel. This removes 99 standalone quantizer dispatches per
50-layer forward while preserving the previous output bytes, improving crossed
512/864 measurements by about 0.3-0.6%. Use
`--use-slower-unfused-int8-inputs` to restore the standalone quantizers.

The fused gated-AdaLN path loads its full 5,376-wide H3 rows as BF16x4 vectors
and writes int8x4. It stages the rounded values locally before computing the
original per-thread RMS sequence, so the reduction tree and every output byte
remain unchanged. Crossed measurements save roughly another 0.1-0.5%. The
existing `--use-slower-unfused-int8-inputs` option retains the portable scalar
and standalone-quantizer fallback.

Q/K RMS normalization and RoPE are performed inside the int8 QKV projection
tile as well. The fused epilogue is byte-identical and improves complete
forwards by 2.1-3.2% at 512 and 1.0-1.8% at 864 in crossed M5 measurements.
Use `--use-slower-unfused-qkv-rope` to restore the separate Q/K kernel.

That epilogue processes four adjacent Q/K dimensions per work item with
BF16x4 loads and stores. The per-element arithmetic and BF16 rounding order are
unchanged, while crossed cool-state measurements improve complete forwards by
about 0.4-1.0% at both 512 and 864. The same
`--use-slower-unfused-qkv-rope` option restores the scalar standalone path.

At up to 2,048 rows, the exact RMS loop uses BF16x4 loads followed by four
explicit ordered FMAs. This preserves every output bit and improves 512-class
forwards by another 0.5-0.6%; larger shapes retain scalar loads because the two
forms tie there. Use `--use-slower-scalar-qkv-rms` to force scalar loads.

The int8 attention-output projection caches its 128 row and column scales in
1 KiB of threadgroup memory instead of rereading them for every cooperative
fragment element. Above 2,048 rows the fused QKV kernel uses the same idea and
then recycles that storage for inverse RMS values; smaller QKV shapes retain
direct loads because the two forms tie there. Both are byte-identical and
improve complete forwards by about 0.2-0.7% where selected. Use
`--use-slower-uncached-int8-scales` to restore direct device-scale loads.

For sequences of at most 2,048 rows, the H3 attention-output projection also
compiles its 7,168-by-5,376 shape into the TensorOps kernel. The result remains
byte-identical while saving about 0.2-0.8% in crossed complete 512-forward
measurements. Larger sequences retain the dynamic-shape kernel because the
specialization regresses there. `--use-slower-uncached-int8-scales` restores
the general dynamic, direct-scale-load implementation.

FC1 also uses an H3-specialized, compile-time 5,376-wide TensorOps loop. It is
byte-identical to the generic loop and saves about 0.1-0.4% in crossed complete
forwards. Use `--use-slower-dynamic-fc1-k` to restore the runtime-bound loop.

```sh
./h3 --profile -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 50 --reuse 1 -o outputs/fox-int8.mp4
```

Use `--use-slower-bf16-mlp` to force the portable close-reference MPS/BF16 MLP
path for numerical comparison. Older Metal hardware selects that path
automatically when the required native TensorOps kernels are unavailable.
For FC2 activation quantization, sequences of at most 2,048 rows use an exact
128-thread reduction. Each thread retains its eight BF16 input values while
computing the group maximum, avoiding a second device-memory read when it emits
the int8 values; crossed M5 measurements improved complete 512 forwards by
about 0.2-0.8% without changing any output byte. Larger sequences retain the
measured 256-thread kernel. `--use-slower-grouped-quantizer` forces the latter
at every size for A/B comparison.

The native baseline targets the original `FL2VA/` and `Ref2VA/` checkpoint
trees. One-shot generation loads and releases phases separately. Interactive
session caching intentionally keeps compatible DiT weights and the video decoder
resident while rebuilding conditioning; peak memory therefore depends on the
retained model, reference workload and workspace. `!cache clear` releases them.

## Native content controls and machine execution

This section documents the current fork, not an unmodified upstream snapshot. Ordered references and fixed target-time guides are separate inputs. References provide subject/style/motion/voice evidence; guides inject encoded conditioning rows at a target time. Neither guarantees identical decoded pixels, a copied soundtrack, voice identity or seamless motion.

### Timed guides and public C calls

The CLI accepts `--guide-image FRAME PATH`, `--guide-clip FRAME PATH`, `--guide-audio FRAME PATH`, and `--guide-clip-audio FRAME PATH`. `--guide-source-start N` and `--guide-source-frames N` modify the preceding guide's source span; `--guide-audio-path PATH` pairs external audio with the preceding clip. Clip guides require an explicit valid `17k+5` span; invalid lengths, out-of-target placement and same-stream overlaps are errors, not hidden trimming or last-wins precedence. Source selection can address a bounded clip inside a longer source. Native frame indices are resolved nonnegative target pixel-frame indices; the Python planner resolves end-relative indices before execution. Interactive equivalents are `!guide-image`, `!guide-clip`, `!guide-audio`, `!guide-clip-audio`, `!guide-span`, `!guide-audio-path` and `!guides [clear]`.

`h3.h` exposes `h3_guide {kind,path,audio_path,frame_index,source_start_frame,source_frames,include_embedded_audio}` and `h3_params.guides/guide_count`. First/last convenience fields remain accepted, but execute through guide conditioning. Mixed references require the actual Ref2VA model and matching adapter target; no silent FL2VA fallback is performed.

Additional synchronous public calls are:

```c
h3_result *h3_resume(h3_ctx *, const char *prompt, const h3_params *);
h3_result *h3_encode_media(h3_ctx *, const char *input_path,
                          int stream_policy, const h3_params *);
h3_result *h3_decode_latent(h3_ctx *, const char *input_path,
                           int diagnostic_preview, const h3_params *);
```

`h3_encode_media` stream policies are `1` video, `2` audio, `3` audiovisual. Set `latent_output_path` for encode and `output_path` for decode. Video encoding requires the requested frame count and explicitly converts encoder `ceil(T/4)` time to generation cadence using endpoint-aligned linear interpolation; this is marked in the latent metadata, not a claim of lossless temporal equivalence. Standalone audio encode uses an explicit 16-second decode bound; it is distinct from Ref2VA references' 2–15-second/aggregate15-second policy. Audiovisual encode aligns audio to the requested video duration. An audio-only latent need not invent a video stream. Generating an audiovisual target from initial latents still requires compatible target streams/shapes.

### Latents, partial denoise, masks and checkpoints

`h3_params` adds `initial_latent_path`, `mask_path`, `start_transition`, `resume_path`, `checkpoint_path`, `latent_output_path`, `request_digest` and `on_transition`. Start transition indexes the **original full dual-stream sigma schedule**; initial clean latents are noised at each stream's own sigma before controlled sampling. Terminal start is decode-only. A new edit resets solver history; exact resume restores it and does not permit a changed prompt, conditioning, model/adapters, shape, schedule or execution configuration. Native execution derives its identity as well as using the supplied digest; repeating a digest cannot authorize a different computation.

Exact-controlled requests hash the selected model components (including the video VAE normalization sidecar), conditioning media, shader and executable by content, alongside the immutable adapter snapshot and execution configuration. Hashes are cached per context against device/inode/size/mtime/ctime; the first controlled request reads those dependencies, while unchanged warm requests reuse their hashes. Resume validation precedes encoder/DiT preparation. Changed dependencies during preparation/sampling fail before checkpoint publication. Older metadata-only checkpoint identities are intentionally incompatible; the file envelope is unchanged.

Native files are versioned envelopes, **not safetensors/Comfy latent interchange**: clean latent type5; solver state type4. `h3_latent` records video latent dimensions/count, audio latent length/count, optional absent streams, source frame count and temporal mapping (`0` generated/native, `1` encoder-linear-endpoints). Video is channel-major `[24,T,H/16,W/16]`; audio `[32,2,T]`. Payload is native-endian IEEE float; incompatible representation, malformed lengths/shapes and nonfinite tensors are rejected. Do not reinterpret raw encoder temporal output as generated cadence or import foreign normalization based on filename.

Masks use the same native latent container, with per-element finite weights: **0 preserves, 1 edits**. Protected values follow fixed clean/noise trajectories at their own stream's sigma; soft-mask projection occurs once per completed proposal, not again on model predictions. RES history keeps raw clean predictions. Fully protected terminal **latents** match clean preservation values; VAE decoding still need not reproduce original source pixels/audio exactly.

Controlled stateful execution requires full50layers, denoise reuse1, core reuse1, no token reduction and no SSD streaming; unsupported approximation combinations fail instead of silently dropping cache state. This is CPU sampler orchestration around native model evaluation, not a faster-path claim. The existing noncontrolled GPU path may round differently; compare interrupted/resumed execution to an uninterrupted run of the **same controlled path**.

`h3_transition_callback` returns0 continue,1 pause,2 cancel at completed transition boundaries. A pause must publish a valid checkpoint before returning paused. `h3_result.status` is0 succeeded,1 paused,2 cancelled; a non-null result is not automatically a completed video. `diagnostic_preview` distinguishes explicitly requested nonterminal checkpoint decoding from a final clean result. Errors return null and are available through `h3_last_error`, `h3_last_error_code`, and `h3_last_error_field`. Callers must inspect status before final-output publication and free results normally.

Performance execution modes are orthogonal to quality/precision: task.execution_mode overrides profile.execution_mode then administrator execution_mode, defaulting to `throughput`. Resolved plans/jobs/inspect expose execution_mode, execution_reason and controls. Throughput omits native control/checkpoint, retains the ordinary supported native route, and offers process-abort cancellation without a checkpoint or partial-output publication; pause/resume return an actionable repair to prepare a resumable task. Resumable adds boundary control/checkpoint and requires full layers/reuse1/no reduction/no SSD. Initial-latent/mask/partial edits automatically select resumable; explicitly requesting throughput for such an edit fails. Native transition callback is installed only for an actual control path. Human CLI without controlled options retains its normal route. No speed percentage or warm-throughput parity is established by this contract. Exact-resume eligibility does not mean all-BF16 arithmetic or better perceptual quality.

The AI queue worker starts a **new native process per job**: `Tasks.worker` calls `execute_native`, whose `subprocess.Popen([engine,"--manifest",path],...)` launches execution. Media-source encoding is another subprocess. Native resident weights/context/conditioning are not shared across jobs, so repeat tasks repay loading, adapter merge and quantization despite possible warm filesystem pages. Existing human interactive caching remains available and can have materially different warm latency. Process-cold benchmarks cannot certify optimal resident throughput; a persistent native worker is not implemented or promised here.

### Strict manifest and events

`./h3 --capabilities-json` performs no model loading and emits schema1 mechanism availability with `model_inventory_probed:false`, `requires_model:true` and quality evidence. This probes implemented interfaces, **not installed-weight availability or quality acceptance**. `./h3 --manifest PATH` is a standalone invocation; do not combine it with ordinary flags. It validates a bounded strict JSON document and rejects unknown/duplicate/malformed fields.

Required manifest fields: `schema_version:1`, `operation` (`generate`, `resume`, `encode`, `decode`), `model_dir`, `target` (`fl2va` or `ref2va`), `request_digest` (64hex digits), and `params`. Optional top-level fields are `prompt`, ordered `references`, `guides`, `loras`, `control_path`, and `checkpoint_path` (or put checkpoint_path in params, not both). References contain `kind`, `path`, optional `audio_path/include_embedded_audio`; guide entries additionally contain `frame_index` and optional source span. LoRAs contain `path,strength`.

```json
{"schema_version":1,"operation":"generate","model_dir":"/models/H3","target":"ref2va","request_digest":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","prompt":"A person walks forward.","params":{"width":512,"height":512,"frames":22,"steps":8,"sampler":"res_multistep","scheduler":"simple","video_shift":12,"audio_shift":3,"dit_layers":50,"denoise_reuse":1,"core_reuse":1,"output_path":"/outputs/result.mp4","vae_precision":"f32"},"references":[{"kind":"image","path":"/assets/person.png"}],"guides":[{"kind":"image","path":"/assets/opening.png","frame_index":0}]}
```

This is a wiring-sized request, not a checkpoint-quality recommendation. `params` supports the native geometry/sampling/execution fields and latent/control paths. Encode/decode use `input_path`; encode supports integer `stream_policy`; decode supports integer `diagnostic_preview`0/1. `vae_precision` is `f32` or `bf16`. Generate/resume manifests require resolved `17k+5` frame count and at least one requested media/latent output; encode requires a latent output and decode a media output. `preview_denoise` is not accepted in machine execution. Use capabilities and administrator profiles rather than asking a small model to invent kernel flags.

Stdout is versioned JSONL: progress events followed by terminal `result`, `paused`, `cancelled`, or `error`; stderr retains diagnostics. Terminal output fields are `media`, `latent`, `checkpoint`, with null for unpublished artifacts. Success checks requested regular output files before reporting completion. `error` includes structured code/field/message. An optional control file contains exactly `{"action":"run"}`, `{"action":"pause"}`, or `{"action":"cancel"}`; control requires checkpoint output configuration. An unreadable/malformed control file is an error, not a successful cancellation. Plan/job persistence is provided by the Python module below, not by this one-shot manifest runner.

`./h3 --schedule-json SCHEDULER STEPS VIDEO_SHIFT AUDIO_SHIFT` is a model-free CPU query returning schema1 `requested_steps`, effective `transitions`, and `video`/`audio` sigma arrays including terminal zero. These are the scheduler values before any sampler-specific first-sigma adjustment. The planner uses this native query for beta/beta57 partial edits rather than reimplementing beta inversion in Python. For 1,000 requested steps, beta has847 transitions and beta57 has815; terminal clean decoding uses that effective boundary, and larger start indices are rejected before admission.

## AI task interface

`tools/h3_tasks.py` is a Python-standard-library durable local planner and serialized worker. It is not a hosted model, HTTP server, official Context-IR implementation or guarantee of generation quality.

```sh
python3 tools/h3_tasks.py capabilities --json --root /absolute/task-state --config /absolute/admin.json <<<'{}'
python3 tools/h3_tasks.py profiles --json --root /absolute/task-state --config /absolute/admin.json <<<'{}'
```

Every operation consumes one JSON object on stdin and returns one `{"ok":true,...}` or `{"ok":false,"error":...}` object. `--root` is required; use the same root/config across prepare, submit, worker and inspect. Administrator config supplies absolute `engine`, shared device identity (e.g.`metal:0`), optional ffmpeg/ffprobe and a `profiles` dictionary. Each profile supplies `model_dir`, explicit `target`, native `params`, `max_frames`, evidence/exactness metadata and **complete** dependency entries `{role,path,sha256}` for selected weights, encoders, decoders, adapters, tokenizer/config and shader files. No operational production profile or dependency hash is invented by the tool. Every configured adapter must have a pinned dependency. The native executable and relevant execution environment are pinned separately; changed dependencies fail before launch. All local roots addressing the same GPU must use the same configured device ID so the worker lock serializes them.

### Prepare, submit and inspect

1. `asset.register` accepts `{"path":"/absolute/source.png"}` and copies/probes pinned content into the managed content-addressed store. Subsequent task requests use opaque `asset_id`s rather than runtime paths. Native clean latent/checkpoint files are strictly probed; arbitrary files are not accepted merely because of an extension.
2. `prepare` accepts `{"task":{"profile":"configured-ref-v1","prompt":"...","frames":124,"references":[{"asset_id":"a_...","role":"image"}],"guides":[{"asset_id":"a_...","kind":"image","frame":0}],"outputs":{"media":true,"latent":true}}}`. References are ordered. Planner resolves assets/profile, end-relative guide positions, normalized media transforms and compatibility before creating an immutable `plan_id/plan_digest`; it does not run inference. Inspect `summary`, `resolved`, transformations and quality warnings before submission.
3. `submit` accepts only `plan_id`, `plan_digest`, and a nonempty `idempotency_key` (plus optional operation discriminator). Same key+plan returns the same job; same key with another plan fails. Submission queues work; it does not start an invisible background service.
4. Run `worker` explicitly with the same flags and stdin `{}` to drain queued work (`{"once":true}` requests one job). Only one worker owns the device. It executes native manifests with argument arrays, not shell strings, and publishes requested validated outputs as asset IDs.
5. `inspect` accepts `{"plan_id":"p_..."}` or `{"job_id":"j_...","cursor":0,"limit":50}`. Event pagination is bounded (maximum200); job inspection reports state, output/checkpoint IDs, failures and permitted next operations. Worker interruption yields `interrupted`, never an automatic expensive replay.

### Normalization and structured intent

Profiles default to versioned `standard_media_v1`: declared guide cover-crop, **reference_resize** as a separate native-reference operation, target-time resampling, audio32kHz stereo conversion and necessary right-padding. Prepare discloses actual operations and pins policy contents in the plan digest; submitting that digest approves that specific prepared execution. Optional `normalization:"require_exact"` rejects required conversions. Semantic `reference_trim`, `reference_frame_floor`, `reference_audio_trim`, and `reference_first_frame` are excluded and need explicit approval, as do overrun trim, clip reinterpretation, source selection, dropping conditioning, speech changes and internal downrender. Routine conversion policy does not silently authorize content changes.

`guide_first_frame` is likewise an explicit semantic transformation: using multiframe video as an image guide requires approval even when its canvas already matches. Profile `params.reference_image_size` uses the native manifest strings `"match"` or `"max"`, not integer enum values. Preparation enforces native reference limits (12 total; at most nine images, three videos and three audio inputs, counting paired references in both modalities) and target/reference compatibility.

Registered media declares `media_policy:"self_contained_v1"`, an explicit demuxer, file-only protocols, canonical `frames_24` and decoded `audio_samples_32k`. Playlists and MOV external data references are rejected before probing rather than fetched implicitly. Resolved plans expose `reference_preprocessing` and `guide_spans`, including actual native-reference resize/stretch behavior, source selection and audio counts. Guide source frame indices use the canonical24fps clock **after resampling**, not the container's original frame index. This metadata makes reference preprocessing inspectable; it is not a reference-fidelity guarantee.

Optional `task.intent` replaces freeform `prompt` with deterministic structured formatting. Fields are `setting`, `subjects:[{id,description,reference_ids}]`, `actions` (text or list), `camera`, `dialogue:[{speaker_id,language,text}]`, `reference_use:[{asset_id,role,relationship,use,subject_id}]`, `ambience`, `music`. A voice binding uses `use:"voice",subject_id:"person"` for an audio-bearing reference and a defined subject; dialogue speaker_id names that subject. The formatter emits the six plain-text `name:` sections from the official reference guide, stable `<Subject N>`, independently counted `<Picture N>/<Video N>/<Audio N>`, and `<d>[Chinese] …</d>` dialogue. Do not put reserved labels/delimiters inside plain structured text. It does not inspect assets, transcribe audio, infer unseen identity features or perform general semantic contradiction detection. Multiple views of one person can share one subject. Freeform input remains possible but does not receive fabricated structured guarantees.

### Revision, continuation and control

- `revise` creates a **new edit plan** from a job's published clean latent/media or a registered asset. Patches may change prompt/intent, references, guides, mask, start_transition and normalization, not immutable profile/shape. Registered-asset revision supplies explicit task profile and target frames. Media sources are genuinely encoded before partial-denoise generation; a revision is not an exact resume.
- `continue` accepts job_id or asset_id, `tail_frames` (valid `17k+5`,default22), positive `additional_frames` divisible by17, explicit `audio_mode` (`condition`,`silent`,`discard`), `join_mode` (`new_segment`,`concatenated`), `assembly_policy:"explicit_reencode_no_crossfade"`, and a task containing the profile/prompt. It owns target frames and tail guides. Source must be24fps; concatenation requires matching canvas. The prepared assembly reports overlap, generated overlap discarded, final length, audio policy and join time. It is explicit re-encoding without hidden crossfade, not perfect or bit-exact generated continuity.
- `pause` and `cancel` accept job_id; only resumable jobs can pause or retain checkpoints (`keep_checkpoint:true`). A never-started paused resumable job has no solver checkpoint; inspect reports a fresh submit route. Running resumable control is observed at completed transitions. Throughput cancellation aborts its worker-owned native process group and publishes neither partial outputs nor a checkpoint; asking to pause or keep a checkpoint produces a repair to prepare a resumable task. Repeated terminal control is idempotent.
- `resume` accepts **only job_id** (plus operation discriminator) for a paused/interrupted/cancelled job with a durable checkpoint. It creates a new immutable prepared resume plan, to submit normally with its digest/new idempotency key. No changed intent/profile is accepted. Exact computational continuation is same-build/device/configuration behavior, not a guarantee of original source pixels, identical speech waveforms or cross-backend bitwise output.

Assembly is generated only by `continue`; caller-supplied `task.assembly` is rejected. Job-based revision retains the parent's resolved profile and default normalization-policy snapshot while rechecking pinned dependency availability. When continuation published concatenated media, revision encodes that complete published clip rather than its unassembled segment latent. If only a clean latent was requested/published, revision explicitly retains that generated-segment length; no assembled media is invented.

### Evidence boundary for this integration

The following records describe the pre-beta5 controls/task integration. The later beta5 implementation and author-weight proof are recorded separately in [Eros beta5 setup](EROS_BETA5_SETUP.md); historical executable identities and counts below must not be relabeled as the latest beta5 build.

Parent-reported final native acceptance now passed: clean compilation;1793host checks; sampler and GPU suites covering LoRA, reductions, BF16, AudioVAE and mux; real3-second standalone audio independent of22video frames; terminal callback/result and512media-versus256standalone handling; zero-mask full native equality; partial denoise; all three samplers' pause/resume byte equality and changed-prompt/same-supplied-digest rejection; and RefTurbo8 with M5rowFC2 plus mixed references/guide (observed54.613seconds for that case). Earlier mixed/timed guides, bounded selection from a600-frame source, AV/video-only encode/decode and eight malformed manifests passed. The audio-only bound fix has therefore been rerun successfully; it is no longer pending. Current native executable SHA256 is `d6209fa2212dee29f91c4d3c497beceb05e4302c8fd1bc11f8d817992daa0724`.

The54.613-second mixed Turbo case establishes only schema/weight/LoRA+rowFC2 inference compatibility. Its harness used a blue-jacket adult/park guide but explicitly prompted a red fox; the output fox follows that conflicting prompt. Retain this contradictory control: it is **not** first-frame, identity or voice-adherence evidence. A separate aligned comparison subsequently generated both reference-image+reference-audio with a frame-zero guide and the same request without that guide:512×512,107frames,8-step RefTurbo,RES/simple,F32VAE,full50layers,reuse1,seed42. Both MP4s decoded successfully; visual inspection showed the referenced clothed adult and park with the requested wave. This demonstrates the exercised combined path, not exact source pixels, voice identity, lip sync, or a controlled quality improvement.

Parent's two-alternated-cycle benchmark used512×512,39frames,4Euler/serving steps,full50layers,F32VAE,process-cold with filesystem cache not flushed. Median full wall seconds: old43.0846785,new humanCLI39.7442995,new machine throughput41.1369515,controlled40.984869. Old/newCLI/newmachine output MP4 hashes agreed; the two controlled outputs agreed with each other but differed from that fast-path hash, consistent with separate arithmetic paths—not a quality comparison. Two samples do not establish a general speedup, significant-regression verdict or resident warm-throughput parity. Latest detailed commands/hashes/results belong in the parent-owned verification record.

Final Python planner/asset regressions against the real isolated native binary ran39tests:37passed and2inference tests skipped by that suite's explicit opt-in gate. Separate real-model task acceptance then passed prepare/submit/idempotency/inspect, actual media-only source encoding and revision, transition-boundary pause and resume, concatenated continuation, throughput generation without checkpoints, and running throughput process-group cancellation without published outputs/checkpoints. Changed-intent exact resume was rejected. These were native model executions, not fake binaries. They do not certify subjective quality, voice identity, lip-sync, long-duration stability, every approximation/checkpoint combination, or seamless generated continuity.

Observed artifact records are preserved under `/Users/mics/Movies/H3-Controls-20261026/`: `native-acceptance-summary.json`, `performance-summary.json`, `task-acceptance-summary.json`, and `aligned-guide-evidence/` (both videos, manifests, logs and summary). The isolated remote run root is `/Users/tedmacbookpro/projects/h3-lora-dev-20261025T185924Z`, with `acceptance/`, `performance/`, `task-acceptance/` and `aligned-guide-evidence/`. The records retain the earlier audio-only failure alongside its passing fixed rerun, and identify per-case native executable SHA-256; the final pre-beta5 executable is `d6209fa2212dee29f91c4d3c497beceb05e4302c8fd1bc11f8d817992daa0724`. Subsequent beta5 extensions require separate verification; these results must not be relabeled as beta5 evidence.

## Eros Max beta5 runtime

The isolated beta5 runtime uses the author's full BF16 TURBO-hybrid source, not a renamed official transformer or an emulated ConvRot INT8 checkpoint. Native schedule preparation recognizes rank8 AdaLN/time-table curves, keeps block modulation at BF16 and final modulation/output heads at F32 with a BF16 RMSNorm boundary. Existing full-width final-head fast paths remain unchanged. Required and optional transformer inventories share discovery; optional Ref2VA accepts valid single-file checkpoints without requiring an index, while malformed/unreadable inventories fail closed.

Eight pinned task presets cover text/reference × ER-SDE/beta57, RES/simple, LCM/simple and Euler/simple, without extra Turbo LoRAs or approximation caches. Real author-weight reference/image/audio/guide generation, exact native ER resume, installed-profile preparation and task pause/resume were exercised. Source/converted hashes, numerical reference commits, tested scopes, artifact paths, licensing boundaries and the preserved failing-before discovery case are in [EROS_BETA5_SETUP.md](EROS_BETA5_SETUP.md). These are implementation and exercised-path results, not universal quality, voice/lip-sync or commercial-license certification.

## Fork-wide review corrections (2026-10-26)

Review baseline: local upstream commit `8974cc055ea9c02fcd14cc27dfda3e1027c05153` through `6256272`, eight fork commits and 69 changed files. Independent standards/specification and engine/numerics/weights/CLI reviews covered the change inventory; this is not a claim that every model/device combination was executed.

Corrections preserve CLI FL2VA first/last-frame presentation and aligned endpoint placement; promote clip guides when attaching external audio; compare guide adjacency on the integer frame clock; authenticate controlled dependencies by content, including the consumed VAE normalization sidecar; and validate checkpoints before inference preparation. Task preparation now uses native reference-sizing strings and modality limits, rejects public assembly injection, requires explicit first-frame selection for multiframe image guides, preserves revision profile/policy snapshots, and revises published assembled media rather than its segment latent. Latent-only continuation revision retains segment semantics. Beta/beta57 partial and terminal starts use native effective transitions. Reduction-test oracles explicitly reject nonfinite values.

Isolated verification root: `/Users/tedmacbookpro/projects/h3-lora-review-20261026`; generated evidence is under `outputs/review-acceptance/`. Build and exercised suites passed: 5,293 host checks; sampler; LoRA; Metal reductions; BF16 VAE projection; AudioVAE primitives; Torch curve fixture; seven converter tests; and 46 planner/asset tests (the two explicit model-inference tests were skipped in that suite invocation).

Real beta5 ER-SDE generation, transition-boundary pause and resume passed. Uninterrupted/resumed clean-latent SHA-256 both equal `30a41f2aa23a6db12274801870f3839fe80ebb4ec7cb43f9f1aa34b8d66d89aa`; MP4 SHA-256 both equal `6c7102c015f5b4e31085af7e0dcac31991886dc38b52759a21ebaca69adb808a`. Equal-length reference-audio and normalization-sidecar mutations with restored mtime were rejected as `checkpoint_incompatible` with zero progress events. This rejection still incurs model/dependency discovery and content hashing; it is not a claim of zero startup cost.

The one-shot first/last-frame CLI smoke requested 24 frames, produced 39 aligned frames, and exercised the vision phase. External-audio guide modifiers passed one-shot admission/rejection and interactive state checks. Native schedule queries returned 847 beta / 815 beta57 transitions for 1,000 requested steps; terminal starts passed the schedule guard, and past-terminal starts were rejected. These results establish exercised behavior, not subjective visual/voice quality, universal cache parity, or production deployment.

Both opt-in real task cases subsequently passed: owned-process throughput cancellation published neither outputs nor checkpoint; continuation publishing both media and latent revised the complete 39-frame assembled media, encoded it, and completed the controlled revision. The revision test's explicit approval list was corrected to retain time rounding and audio padding for its inherited five-frame audio guide; production approval checks were not weakened. A separate launch from outside the runtime directory failed to locate `h3_shaders.metal`; the passing run used the isolated runtime working directory, as did the original harness. The current CLI/task executable SHA-256 is `9c1cd0b294a96d7859bb9706181ba4b55af7031b2ac324bc4a8693acbf2d2a1e`. It includes the later help-text rebuild; the native pause/resume hash comparison above preceded that help-only rebuild.

