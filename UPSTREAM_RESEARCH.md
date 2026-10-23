# Upstream PR and fork research

This is a source/evidence review; external patches are not merged here. The
comparison is against upstream `8974cc055ea9c02fcd14cc27dfda3e1027c05153`.
The user requested speed optimizations and bug fixes as well as LoRA references.
Upstream h3.c is MIT; retain original author attribution for any future port.
Model-weight licensing is separate. Public performance numbers below are the PR
authors' measurements, not measurements of this fork.

## High-value references

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
