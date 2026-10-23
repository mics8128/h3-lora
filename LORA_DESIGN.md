# Immutable multi-LoRA design

## Provenance and scope

This implementation follows the specification commissioned from the exact
`llm-esapp/gpt-6-astra` model: immutable, additive, load-time composition.
The subsequent Turbo design adds explicit FL2VA/Ref2VA targeting and sampling;
see [TURBO_DESIGN.md](TURBO_DESIGN.md). Text encoders and VAEs are not adapted;
there is no merged-checkpoint export, persistent weight cache or hot swap.
The existing MIT notice is retained; adapter licenses remain independent.

## Public and lifecycle contract

`h3_load_dir(model_dir, specs, count, target)` accepts `h3_lora_spec { path, strength }`
and `H3_LORA_FL2VA` or `H3_LORA_REF2VA`. No-adapter callers pass `NULL, 0, H3_LORA_FL2VA`.
Finite signed strengths are required. Either
signed zero disables the entry without opening, hashing or retaining its path.
The context owns copied active specifications and immutable original-dtype
adapter snapshots; `h3_get_loras(ctx, &count)` returns borrowed canonical specs.
Repeated entries add rather than replace. Composition order is canonical by
content SHA-256 and strength, not CLI order. Identity includes interpretation
version and is appended only to active-stack prepared-DiT cache keys.

Construction validates adapters against the selected actual FL2VA or Ref2VA
metadata store before Metal work. Missing selected bases fail; base tensor
payloads remain unloaded until generation. Weight stores
borrow the context-owned stack and bind only before their first payload load.
DiT binding precedes token refinement and AdaLN precomputation. Cache clearing
frees prepared models but retains snapshots. Seed-only reuse avoids remerging;
prompt/shape/step changes use the same snapshots when rebuilding. Destruction
frees prepared models before the stack. Replacing source files cannot mutate
an existing context; base files must remain unchanged as in the original loader.

Active-stack SSD streaming is rejected, never silently disabled: its direct
slot reader otherwise bypasses adapted weight loading. Active adapter targets
must match request mode. Ref2VA requires a real ordered reference image and
match sizing; first/last-frame FL2VA is not Ref2VA. No-adapter automatic routing
is unchanged.
The CLI, interactive SSD setter, C generation API and internal loader enforce
these boundaries. `!status` reports the fixed stack; changes require a new
context. `--info` snapshots active adapter payloads, unlike metadata-only base
inspection without adapters.

## Composition and supported format

Ordinary rank-2 F32/BF16/F16 A/down `[rank,in]`, B/up `[out,rank]` pairs are supported
through exact recognized native/Comfy prefixes and explicit Kohya aliases.
There is no generic underscore replacement or arbitrary prefix stripping.
For each target:

`W' = store(F32(W) + sum(strength * alpha/rank * row_map(B @ A)))`.

Explicit scalar F32/BF16/F16 tensor alpha wins. For ordinary `networks.lora`
metadata, finite nonnegative scalar `ss_network_alpha` is the fallback; otherwise
absent alpha defaults to rank. Tensor/metadata conflicts are reported, with
tensor precedence; optional `ss_network_dim` does not change scaling. IEEE half
factors widen exactly, including subnormals and signed zero. Destinations remain
F32/BF16 only. Unknown scaling, layout, `ss_network_args` or tensor kinds fail
closed. All active tensors must be accounted for, including omitted layers.
DoRA, rsLoRA, dense deltas, bias/norm updates, convolutional and quantized adapter
factors are outside this contract. Missing pairs, aliases, incompatible shapes,
malformed offsets/overlap and nonfinite values are errors.
`Dynamic` metadata alpha requires per-target tensor alpha or an explicit supported
baked declaration. Known Comfy H3 conversions declaring alpha folded/dropped into
B (flat baked scales 0.125 or 1.0), and the exact normalization declaration
`lora_up := lora_up * (alpha / rank); alpha tensors removed`, retain their stored
factor values: baked scaling is never multiplied again. The recognized layout is
`comfyui_minimax_h3`; unknown baked scaling declarations and simultaneous scalar
metadata alpha plus baked scaling fail closed.

CPU Accelerate SGEMM uses bounded 128-output-row and 64-rank panels. Factors
remain original dtype; scratch is bounded rather than a full F32 matrix or full
checkpoint copy. Each element accumulates all contributions in F32, adds the
base once, then receives one BF16 round-to-nearest/ties-to-even conversion (or
stays F32). The official base mapping is never mutated. Existing int8 selection
runs afterward and regenerates scales from adapted weights. No adapter executes
inside denoising. CLI-order identity is backend-local, not a cross-Accelerate
version or cross-device inference guarantee.

## Verified source layout

Comfy H3 splits fused QKV into contiguous Q/K/V projections:
[Comfy H3 model](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/model.py#L176).
Native `h3_shaders.metal` consumes head-interleaved QKV. With 56 heads and
128 dimensions, source row `(q*56+h)*128+d` maps to native
`(3*h+q)*128+d`. The compositor gathers B rows only; A columns and official
base rows are unchanged. Token-refiner QKV uses the same mapping.

Comfy's fused FC1 is gate then up:
[Comfy SwiGLU](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ops.py#L1008-L1010).
Native `h3_shaders.metal` also consumes gate in the first half and up in the
second, so these fused FC1 factors need no permutation. These sources establish
layout semantics; no source implementation is copied into this MIT project.

## Exact candidate header evidence

Complete local artifacts were hashed by the parent; these facts are distinct
from successful production loading, numeric tests and generation acceptance.

| Candidate | Exact version/file | SHA-256 |
|---|---|---|
| [Unlocked](https://civitai.com/models/2925727) | 3413468 / 3305979 | `a5fcea727e52394427547dec8d1217e508298c58a79fdc95e647ccdd9ff86ac6` |
| [Turbo](https://civitai.com/models/2837571) | 3202732 / 3091422 | `4f95afb2ab503eac876ae5571f4d78b0e0fd0198b626a89f4b7727018ed198a9` |

Unlocked has 600 tensors: 200 rank-16 F32 down/up pairs and 200 alpha tensors
(alpha 16), covering QKV, attention output, FC1 and FC2 in all 50 core blocks.
Names use `lora_unet_blocks_<n>_...`; no token-refiner updates are present.

Turbo has 416 tensors: 208 rank-64 BF16 A/B pairs, the same four matrices in
50 core blocks and two token-refiner blocks. Names use `diffusion_model.`.
Alpha is absent; its metadata explicitly specifies `W_eff = W + B@A`, so the
ordinary alpha=rank policy gives effective scale equal to strength.
Its metadata states that 51 AdaLN pairs were removed due to incompatibility
between full input width 2688 and a pruned width-8 model. The remaining ordinary
full-sized matrices are a compatible subset by header, not proof that native
Euler inference reproduces four-step distillation quality. No missing AdaLN
updates are reconstructed, padded or silently skipped.

## Acceptance and publication boundaries

For this implementation workflow, the local machine is for source/documentation
editing and Git management only; its memory and disk cannot support model work.
Keep only source, documentation, tiny synthetic fixtures and small header
inventories locally. Run all compilation, tests, adapter loading, real-model
numeric probes and inference on the designated remote machine in an isolated
checkout, preserving its existing deployment. Keep model files and generated
media outside the publishable checkout. Transfer adapters to remote storage and
verify their complete SHA-256 values before removing redundant local payloads.
The tutorial commands describe the capable execution machine, not permission to
build or run models on the source-management machine.

Header compatibility is not end-to-end adapter support. The parent must record
actual sizes and runtime match counts, run production synthetic numeric tests,
independent real-adapter sample-panel comparisons (including QKV boundaries and
shared targets), weight-store-to-Metal projection and int8 seam tests, then
native baseline/A/B/A+B benign-video matrices in BF16 and normal M5 modes.
Common comparison settings are 512x512, 39 frames, eight Euler steps, 50 layers,
reuse/core-reuse one, seed 42, no token reduction, references or SSD streaming.
Fully decode and inspect all clips; report generation success, numerical
correctness, visible effect and quality claims separately. Cache smoke must
cover seed reuse, prompt rebuild, cache clear, changed source snapshots and
rejected SSD/target-mismatched requests; no-LoRA/all-zero and existing SSD behavior remain
separate regression gates. Layer thinning is an explicit approximation, never
a reason to leave adapter tensors unvalidated.

Both exact artifacts above have now passed production loading individually and
together. On an M5 Max with 128 GiB unified memory, the baseline/A/B/A+B matrix
completed in both resident BF16 and normal M5 modes: eight clips, each 512x512,
39 frames (1.625 seconds), eight Euler steps and seed 42. A strength of 0.5 and
Turbo strength of 1.0 were used. Every clip fully decoded to H.264 video and
32 kHz stereo AAC; beginning/middle/end frames showed fully clothed adults
walking and waving in a park. All 39 combined-default frames were reviewed.
This proves benign generation, not general style fidelity or Turbo quality.

The exact artifact sizes are 298,257,970 and 620,285,592 bytes. All 200 explicit
alpha values in the first adapter are 16; the second has no alpha tensors.
Together they retain 918,543,562 snapshot bytes and target 208 unique matrices.
An independent float64 scalar reference checked 144 production-merged BF16
coordinates across core QKV, FC1, FC2 and token-refiner QKV, including Q/K/V
boundaries and shared targets: zero mismatches after final BF16 rounding.
Synthetic numeric, strict-parser, immutable-snapshot, canonical-order, unaligned
F32 panel, Metal projection and M5 quantization tests passed remotely. Existing
`make test` ran 1,768 host checks, AudioVAE Metal primitives and the media mux
smoke; optional external fixture-dependent tests reported skips.

The initial FL2VA-target lifecycle smoke passed: seed-only prepared-DiT reuse, prompt-triggered
rebuild, cache-clear retaining the stack, a new context with a new strength,
and SSD/ordered-reference rejection before encoding. Source-file removal after
snapshot is covered by the synthetic fixture. The reviewed code also avoids
resnapshotting an identical repeated path and resolves aliases without repeated
semantic inventory reconstruction.

Observed single-run process metrics (not cold-boot or speedup benchmarks):

| Execution | Stack | Wall seconds | Peak physical footprint GiB | Max RSS GiB |
|---|---|---:|---:|---:|
| M5 default | None | 45.24 | 18.81 | 12.13 |
| M5 default | A | 62.96 | 21.05 | 34.53 |
| M5 default | Turbo | 64.72 | 21.35 | 36.90 |
| M5 default | A + Turbo | 70.60 | 21.13 | 37.28 |
| BF16 diagnostic | A + Turbo | 82.50 | 37.80 | 37.12 |

All eight processes reported zero swaps. Metal DiT tracked peaks were 18.668
GiB in default mode and 36.395 GiB in BF16 diagnostic mode. Physical footprint,
RSS and tracked Metal storage have different accounting semantics; do not sum
them or interpret RSS alone as total workload memory. Adapted weights use new
buffers instead of mutating file-backed originals, which can increase RSS.
The wall-time figures include loading/composition and are not evidence of
Turbo acceleration. The Turbo artifact's missing-AdaLN warning still applies.

The reviewed implementation also generated a combined-stack 243-frame clip
(10.125 seconds) at 512x512 and eight Euler steps. It fully decoded, and
representative frames retained the requested fully clothed park scene. Process
wall time was 305.83 seconds, maximum RSS 36,015,472,640 bytes, peak physical
footprint 25,307,546,096 bytes, and reported process swaps zero. These are
distinct accounting metrics, not additive memory totals.

Decoded RGB video and decoded audio hashes were identical between the unchanged
upstream executable, this fork without adapters, and this fork with a disabled
missing-path adapter (`-0` strength), for the same 256-square/22-frame/four-step
fox scenario. No-adapter SSD streaming also generated decodable media; its
BF16 execution is not expected to match the default M5 int8 bytes. Malformed,
missing, NaN, infinite and trailing-junk CLI strengths were rejected.

No weights, merged weights, model-derived numeric fixtures, generated videos,
credentials, private paths or private assets belong in public commits. Only
synthetic fixtures, generic code and redacted verification records are
published. The model publisher's restrictions do not authorize redistributing
derivatives.
