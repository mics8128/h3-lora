# Upstream PR and fork research

The original source/evidence inventory compares upstream
`8974cc055ea9c02fcd14cc27dfda3e1027c05153`. Following explicit user approval,
selected PR75/PR73 mechanisms were implemented; the dated follow-up below
records current status. Original inventory decisions are historical, not a
second implementation backlog. Upstream h3.c is MIT; attribution is retained
in THIRD_PARTY_NOTICES.md. Model-weight licensing is separate. Author-reported
numbers are not local measurements unless a section explicitly says so.

## Original high-value reference inventory

| Source | Mechanism and reported evidence | Decision for this fork |
|---|---|---|
| [PR 73: VAE precision/cache/prefetch](https://github.com/antirez/h3.c/pull/73), head `aac7b5ab1b920ede22c7ce55e6139d11bde06173` | M5 Max 768x448/124-frame decode reportedly 43.3s F32 to 13.2s BF16, 75.3dB PSNR. Int8 10.5s but visible patch structure and 51.6dB PSNR. | Best speed investigation candidate: benchmark BF16 separately first. Do not silently change decoder precision or adopt the combined cache/prefetch patch as a proven transparent speedup. |
| [PR 75: lazy Metal pipelines](https://github.com/antirez/h3.c/pull/75), head `30b63b51404cae1f2dcc1841abea96f1893c5b49` | Compile on first use, cache per GPU context. Author's M5 Ultra median 58.079s to 49.951s, but private baseline and fixed ordering confound the comparison. | Small useful reference; require same-source randomized startup timing and identical decoded media before porting. |
| [PR 68: preserve VAE cache](https://github.com/antirez/h3.c/pull/68), head `38ccf9b2e17108c09f8b7dd75dc2327a3a3605e4` | Retain decoder when references/anchors change without changing decoder identity. No speed measurement. | Useful interactive-lifecycle improvement, not one-shot speed evidence. |
| [PR 55: attention weight streaming](https://github.com/antirez/h3.c/pull/55), head `8b51e19b9b88ce09e670039aebf8b7e98697e785` | Int8 projection cache and optional MLP streaming; lower memory, about 18GB disk cache, no isolated speed table. | Memory-oriented design reference. Must compose LoRA before cached quantization and include stack identity; do not transplant cache into this immutable loader without invalidation design. |
| [PR 14: folded Turbo](https://github.com/antirez/h3.c/pull/14), head `bb89c68e1ba8df4861149982c6ce9dcd75b93d1a` | M5 Max 960x544 39-frame run reportedly 87s to 65s, 124-frame 8.8 to 6.2min using fewer steps/different weights. Four-step fast-motion smear noted. | Sampling/model compatibility reference, not same-model kernel acceleration. Dynamic multi-LoRA here avoids publishing folded checkpoints. |

## Correctness investigations

- [PR 77](https://github.com/antirez/h3.c/pull/77), head
  `0a95ea10c0cab51d16ef094ee88dba898e23524d`: opt-in noncausal SDPA query
  splitting. M4 Max/macOS15 at 12,256 rows reportedly produced full-frame noise
  without GPU error. Weight-free sampled CPU-reference RMS: 1.0 unsplit versus
  0.00258 split. Every query block still attends to all keys/values. Hardware
  threshold is not proven on M5. The pinned author's weight-free probe was run
  against this fork's unchanged SDPA on the execution M5 Max: 12,256 rows gave
  maximum absolute error `9.574344e-5`, relative RMS `0.001769664`; 32,768 rows
  gave `6.620745e-5` and `0.001697523`. Both passed the probe's CPU-reference
  thresholds with no nonfinite samples. These sampled results do not disprove
  the M4/Ultra reports; no query-splitting workaround is enabled here.
  [PR 64](https://github.com/antirez/h3.c/pull/64) concerns the same class on
  Ultra GPUs and has a higher gate; do not layer both.
- [PR 63](https://github.com/antirez/h3.c/pull/63), head
  `1b5afccec7524c8806597d0837a9aa125995e90b`: one threadgroup barrier between
  reading the reduced maximum and overwriting the same GQA scratch with sums.
  Exact diff shows a write-after-read hazard. Useful minimal correctness patch;
  repeated long-sequence Metal comparisons should precede a port.
- [PR 44](https://github.com/antirez/h3.c/pull/44): causal GQA threadgroup
  memory must be 16-byte aligned. Check the current allocation before applying.
- [PR 56](https://github.com/antirez/h3.c/pull/56): seven H3 dialogue/lyric/
  caption tokens declared only in tokenizer configuration are absent from the
  tokenizer vocabulary. Diff adds those tokens and concrete token-ID tests.
  Relevant for controlled dialogue/audio prompts; inspect both released
  tokenizer files and compare with the authoritative tokenizer first.
- [PR 31](https://github.com/antirez/h3.c/pull/31): FFprobe CSV may emit exactly
  one trailing separator for unselected side data in iPhone MOV. Diff permits
  that separator but still rejects other junk. Relevant to imported references,
  not current generated-reference PNGs.
- [PR 48](https://github.com/antirez/h3.c/pull/48) and
  [PR 74](https://github.com/antirez/h3.c/pull/74): overlapping safetensors
  ranges and JSON string allocations. This fork's LoRA work independently adds
  strict overlap/duplicate validation and bounded encoded-string allocation;
  inspect overlap before adopting another parser patch.
- [PR 7](https://github.com/antirez/h3.c/pull/7): old RES audio-clock fix is not
  compatible with the new pinned packed-AV Comfy contract. Do not reintroduce
  the old slope-based or independent-clock orchestration.
- [PR 11](https://github.com/antirez/h3.c/pull/11) frame-alignment overflow,
  [PR 9](https://github.com/antirez/h3.c/pull/9) nonfinite RGB conversion and
  [PR 4](https://github.com/antirez/h3.c/pull/4) F32 scaled queries remain
  targeted review candidates; no independent reproduction was performed here.

## Fork discovery

The GitHub fork API and PR heads were inspected. Relevant starting points:

- [justwjx/h3.c](https://github.com/justwjx/h3.c): VAE precision/cache work above.
- [jerryfane/h3.c](https://github.com/jerryfane/h3.c): lazy pipelines and parser
  allocation changes above.
- [quantz8a/h3.c](https://github.com/quantz8a/h3.c): GQA/long-attention fixes.
- [maderix/h3.c-ane](https://github.com/maderix/h3.c-ane): alternate ANE route;
  separate execution/backend tradeoffs, not a drop-in M5 Metal speed claim.
- [Shedrackeze002/h3.c-int8](https://github.com/Shedrackeze002/h3.c-int8): int8
  experiments; verify differences from upstream's existing M5 int8 path.
- [QuixiAI/h3.c](https://github.com/QuixiAI/h3.c),
  [neokree/h3.c](https://github.com/neokree/h3.c): discovered forks; no
  independently validated portable speedup established in this research.
- [matrixfede/h3.c](https://github.com/matrixfede/h3.c): CUDA/Linux work is not
  applicable to the current Apple Metal deployment.

No external performance patch was silently merged into the LoRA/sampler
comparison. Keeping that baseline separate prevents attributing a model,
precision or cache change to a Turbo adapter. Any future port requires a
specific reproducer or paired benchmark, license review, attribution and
verification on the execution machine.

## 2026-10-24 follow-up: current optimized fork

The inventory above describes the original upstream comparison. After the user
authorized implementation, PR75-style lazy direct pipelines and the BF16-only
portion of PR73 were ported. PR78 is not applied. Author measurements below
remain distinct from our local measurements; the current same-machine results
are recorded in [COMFY_CACHE_RESEARCH.md](COMFY_CACHE_RESEARCH.md).

| PR / pinned head | Current finding | Decision |
|---|---|---|
| [75: lazy direct Metal pipelines](https://github.com/antirez/h3.c/pull/75), `30b63b51404cae1f2dcc1841abea96f1893c5b49` | Library compilation remains eager; named direct pipelines now build on first use and cache successes per GPU context. All custom direct encoders use the central getter; MPSGraph caches remain separate. | Implemented without arithmetic changes. Paired F32 CLI outputs match complete baseline RGB/PCM hashes. Eager median136.023s vs final lazy136.140s is a noise-scale difference, not confirmation of the author's14% claim. |
| [73: VAE low precision/cache/prefetch](https://github.com/antirez/h3.c/pull/73), `aac7b5ab1b920ede22c7ce55e6139d11bde06173` | F32 remains default; `H3_VAE_PRECISION=bf16` selects BF16 large projections/SDPA with F32 residual/norm/RoPE/SwiGLU/small projections. | BF16-only implemented; invalid values fail and resident decoder identity includes precision. No int8, disk cache or prefetch. Final39f/768p VAE median40.868→13.909s; complete wall136.140→110.814s (18.60% less). These are our workload, not the author's768x448/124f replay. |
| [78: stable tile band](https://github.com/antirez/h3.c/pull/78), `c1459cc89c91fe74d2c9a036bc3e1efbae1c5339` | Auto candidates become 256/272/288/304 instead of also considering320. Author reports about34% less temporal flicker for304 vs320, but **+8% modeled decode cost** and a73f/50step example52→55min. | Quality investigation, not a speedup. Do not change tile and precision simultaneously; local auto maximum remains320. |
| [68: retain VAE across input changes](https://github.com/antirez/h3.c/pull/68), `38ccf9b2e17108c09f8b7dd75dc2327a3a3605e4` | Its objective is already covered by local resident/request separation and removal of reference-change eager clears. Local code additionally retains compatible DiT weights. | Do not copy its unconditional DiT invalidation on input changes; it would lose current reuse. |
| [63: causal-GQA scratch barrier](https://github.com/antirez/h3.c/pull/63), `1b5afccec7524c8806597d0837a9aa125995e90b` | Same max→sum scratch reuse barrier is now present locally, independently exercised by the multi-SIMD numerical oracle. Local visual-encoder GroupNorm also has the corresponding mean→variance barrier. | Correctness fix already covered, not a new speed claim. Keep synchronization. |
| [55: int8 weight-cache streaming](https://github.com/antirez/h3.c/pull/55), `8b51e19b9b88ce09e670039aebf8b7e98697e785` | Disk cache contains prequantized QKV/out/MLP weights, not attention results. Fingerprint does not bind dynamic LoRA order/strength/composition; a model fingerprint mismatch only warns. | Not a default128GB resident-speed strategy. Adapted weights must never be bypassed by a base-only cache. |

### PR75: evidence and integration constraints

The author's installed-deployment study reports three fresh-process samples per
cohort: baseline median58.079s, candidate49.951s, unchanged control57.821s,
with matching decoded RGB/PCM hashes. It used an M5 Ultra,864x480/56f,
8Euler steps and shared parser/compact-QKV changes; it was **not an inference
benchmark of the submitted upstream branch**. Candidate timings varied from
49.123 to57.443s; fresh process does not prove cold OS/Metal caches. The linked
commit-pinned evidence extraction returned404 through both raw and blob URLs
in this review, so only the PR's preserved statements were readable. Do not
quote its roughly14% improvement as independently verified or portable here.
[PR body](https://api.github.com/repos/antirez/h3.c/pulls/75),
[actual diff](https://github.com/antirez/h3.c/pull/75/files).

Lazy creation shifts failures from context creation to first operation. The
getter retains compiler details, but some existing callers overwrite those
with generic dispatch-capacity errors; the PR does not repair all diagnostics.
It adds no thread-safety contract. Current GCD LoRA workers only write
disjoint preallocated shared-buffer ranges and never compile/encode/submit;
GPU encoding remains single-owner per context. A residual eager scope remains:
the int8 MLP caller requests ten pipeline variants before selecting one.
Do not combine a port with unrelated encoder concurrency or caller refactors.

### PR73: numerical contract and quality boundary

Four large projections lower F32 activations/weights to BF16; matmul output
rounds to BF16 **before** widening and adding original F32 bias. Retain F32
residual/norm/RoPE/SwiGLU and small input/output projections. Contrary to its
stale file-header comment, the PR's low-precision path also uses BF16 SDPA by
default; keeping SDPA F32 is a distinct variant, not the reported13.2s path.
Existing generic GPU fallback is not a CPU decoder or automatic recovery to
F32, and the VAE dimensions do not guarantee use of the handwritten M5 NAX
projection kernels. [Pinned VAE source](https://github.com/justwjx/h3.c/blob/aac7b5ab1b920ede22c7ce55e6139d11bde06173/h3_video_vae.c),
[GPU/shader diff](https://github.com/antirez/h3.c/pull/73/files).

Author PSNR uses float RGB in[0,1], not encoded MP4 comparison. BF16 reportedly
changes7–10% of RGB8 pixels by up to4–6levels: visually similar in the author's
face/fur crops, **not bit-exact**. Int8 shows faint16px patch structure. High
PSNR from one latent cannot certify temporal stability or all content. Native
one-second multi-content visual comparisons, float/RGB8 errors and separate
longer flicker/seam checks are required before considering a default change.

The approximately4.8GiB disk cache and concurrent decoder prefetch are separate
startup/I/O/residency mechanisms, unnecessary for testing BF16 arithmetic.
Prefetch overlaps memory with DiT. Keep these out of a minimal precision port.

### PR78: exact scope

The score remains tile-count×tile-area, not a fixed304px tile for all canvases.
Explicit `H3_VAE_TILE_PIXELS` still accepts only256..512 in multiples of16:
the pinned parser cannot choose240 even though the PR discussion includes it.
The reported34% is a specific high-pass temporal-flicker metric comparison,
not a guarantee that latent noise or static checkerboards disappear.
[Pinned tile implementation](https://github.com/bluetorch/h3.c/blob/c1459cc89c91fe74d2c9a036bc3e1efbae1c5339/h3_video_vae.c#L666-L701).

**Implementation boundary:** lazy direct pipelines and BF16-only VAE are now
implemented; F32 remains default. Tile-band changes remain a separate quality
proposal. Existing CLI use is preserved: no cache builders, daemon state,
low-memory streaming or simultaneous tile/default-precision changes. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source attribution.
