# Turbo sampling and immutable model targeting

## Provenance

This implementation follows the completed second design commissioned from the exact `llm-esapp/gpt-6-astra` model (`/tmp/h3-astra-turbo-spec.md`). Algorithms are independently implemented from the mathematical contracts below; ComfyUI GPL source is not copied into this MIT codebase. Model licenses remain separate.

Primary references are pinned to [ComfyUI bc5f880](https://github.com/Comfy-Org/ComfyUI/tree/bc5f880dbdd43e426077f42ccec53bb577ffee9f): `comfy/model_sampling.py` (ModelSamplingAV), `comfy/model_base.py` (MiniMaxH3), `comfy/ldm/minimax/model.py`, and `comfy/k_diffusion/sampling.py`. The RES exponential-integrator primitive is retained from the native host implementation and exercised through the common sampler seam.

## Numerical and lifecycle contract

The public defaults remain Euler / serving / video shift 12 / audio shift 3. Existing default serving GPU arithmetic and per-modality initialization remain on their original path. Default prepared-cache keys gain no new suffix; nondefault sampler/scheduler/shifts and active Ref2VA target identity distinguish prepared models. Adapter content identity remains immutable. History and RNG are run-local, never cached.

`--scheduler serving` supports Euler only and retains its continuous native dual-clock schedule. `--scheduler simple` supports `euler`, `lcm`, and deterministic `res_multistep` (`eta=0`, no CFG++). Simple indexes a 1,000-entry flow table: q=(1000-floor(i*1000/N))/1000, followed by terminal zero. Each modality uses sigma=s*q/(1+(s-1)*q). Six steps distinguishes this grid from the continuous serving grid.

Simple is a different representation, not merely a renamed scheduler. Let c=video_shift/audio_shift and r=c+(1-c)*sigma_video. Carry video Yv=xv and audio Ya=r*xa on the video sigma clock. Before each joint model evaluation recover native audio xa=Ya/r. Native clean estimates are D=x+sigma*v; packed clean estimates are Dhat_v=Dv and Dhat_a=c*Da. Return native audio Ya/c at terminal zero.

Euler advances Y by (sigma_next-sigma)*(Y-Dhat)/sigma. LCM replaces Y with (1-sigma_next)*Dhat+sigma_next*noise; terminal zero returns Dhat without a noise draw. RES uses first/terminal Euler and exponential multistep updates on the same carried representation. Injected velocity/noise seams permit deterministic numerical comparison independently of Metal inference. Native seeded noise is not promised to match PyTorch random arrays.

Simple requires full canvas, 50 DiT blocks, no token reduction, denoise reuse 1 and core reuse 1. Unsupported combinations and nonfinite/nonpositive shifts fail before encoding. DiT stays on Metal; host sampler arithmetic is allowed. Progress, cancellation and previews use the existing generation lifecycle.

## Immutable Ref2VA target

`h3_load_dir(model_dir, specs, count, target)` selects `H3_LORA_FL2VA` or `H3_LORA_REF2VA`; the CLI spelling is `--lora-model fl2va|ref2va`. Active adapters validate against the actual selected transformer schema before payload loading. A missing Ref2VA base fails rather than falling back to FL2VA. Active-stack request mode must match the immutable target, and active SSD streaming remains rejected. Active Ref2VA generation requires an actual ordered reference image path and `match` sizing. Existing no-adapter reference-based routing remains unchanged.

Tensor shape compatibility proves loadability, not training compatibility. Unknown tensor families and incompatible AdaLN updates must fail, never be discarded. Pruned Larry files cannot supply their removed AdaLN updates. Actual safetensors factor dtypes, rather than Civitai metadata, determine decoding support.

## Recommendation record

Source: [Civitai model 2837571](https://civitai.com/api/v1/models/2837571). All rows use CFG 1, simple scheduler, Turbo alone. Numbers below are recommendations selected for the comparison, not measured quality claims. Default shifts marked fallback are adopted Comfy H3 defaults, not per-version best shifts.

| Version ID / version | Representative file ID | Minimum steps | Sampler | Strength | Video/audio shifts |
|---|---:|---:|---|---:|---|
| 3202732 / Larry v4 EMA step600 pruned | 3090387 | 6 | res_multistep | 1 | 12/3 fallback |
| 3391964 / DMAD–Larry full merge, not `_fro` | 3281488 | 6 | lcm | 1 | 12/3 |
| 3382801 / DMAD full conversion | 3271482 | 6 | euler card; separate lcm discrepancy run | 1 | 12/2 |
| 3322352 / Taomate original F32 EMA | 3208044 | 4 | euler | 0.75 | 12/3 fallback |
| 3297123 / Lightx 768 v1.2 | 3181960 | 5 | res_multistep | 1 | 12/3 fallback |
| 3294126 / Lightx Ref2VA | 3179683 | 8 | res_multistep | 1 | 12/3 fallback |
| 3219346 / Lightx 768 v1.0 | 3101219 | 4 | euler | 1 | 6/3 |
| 3219337 / Lightx 8-step v1.0 | 3101211 | 4 | euler | 1 | 12/3 |
| 3206543 / Lightx v0.1 | 3088145 | 4 | euler | 1 | 12/3 |
| 3275758 / Lightx 768 v1.1 | 3159814 | 5 | res_multistep | 1 | 12/3 fallback |

The three older versions have null Civitai version descriptions. Their disclosed fallback is the [original trainer's pinned specification](https://github.com/ModelTC/Minimax-H3-Turbo/blob/02e26d591f7a04d5d1a074c9566d5dd4f22f6225/README.md), with [Euler/simple workflow instructions](https://github.com/ModelTC/Minimax-H3-Turbo/blob/02e26d591f7a04d5d1a074c9566d5dd4f22f6225/COMFYUI_SETUP_AND_INFERENCE.md). The non-768 8-step model says “8 / 4”; four is selected under the minimum-recommended-count rule, not inferred from its filename.

DMAD's Civitai card says 6–8 steps, er_sde/euler and 12/2. Its [original pinned instructions](https://github.com/Yzmblog/DMAD/blob/32ac967324eb0ebba654faeffda99f5def1e86cb/comfyui/README.md) prescribe LCM/simple re-noising and identify four trained steps. Preserve separate six-step card-minimum Euler and six-step LCM discrepancy runs; do not describe Euler as DMAD-equivalent. `er_sde` is not implemented because every selected recommendation has a supported alternative.

The inventory contains ten versions and 21 weights; a representative-per-version suite is not an all-files suite. The remaining eleven files include Larry non-EMA/other checkpoints, merge `_fro`, compressed DMAD/Taomate ranks and resized Lightx v0.1. Their identities, exact hashes, coverage and any failures belong in the execution manifest, not silently omitted or treated as quality-equivalent.

## Evidence boundaries

All compilation, fixtures, header inspection, model loading and generation run on `ted`, never the local source checkout. The comparison runner records exact adapter hashes, base index/config hashes and shard sizes, recommendation provenance, commands, precision diagnostics, reference/prompt hashes and output hashes. Base payloads are not claimed to be hash-authenticated. With `--profile`, native sampling reports the actual resolved video/audio sigma arrays and counted joint model evaluations; the runner requires these to agree with the requested sampler/step count. Numerical state/noise-draw parity is independently checked by the native numerical probe.

Successful loading, sampler correctness, successful generation and subjective quality are separate claims. A single benign seed does not establish a universal winner. Ref2VA is a separately conditioned task. Tiny smoke clips do not replace 124-frame/24-fps quality comparisons at the documented resolution. Final media must be fully decoded and reviewed, with audio listened to, before reporting quality. This document makes no claim that those runtime acceptance checks have passed.

## Observed numerical and lifecycle checks

- The production sampler probe covered 45 cases / 216 intermediate transitions: Euler, LCM and RES; 1, 4, 5, 6 and 8 steps; shifts 12/3, 12/2 and 6/3. Independent execution of the pinned Comfy functions with identical injected native states/noise matched all carried/native states and noise draw counts; maximum absolute difference was `1.7881393432617188e-7` (`atol=rtol=3e-6`). The private oracle is not distributed as MIT source.
- All 21 Turbo artifacts and four additional style artifacts passed 144 independently calculated float64-coordinate checks each: **3,600 coordinates, no mismatches after F32/BF16 rounding**. Coverage includes QKV interleaving boundaries, shared targets, F16 factors, scalar metadata alpha, per-tensor alpha and already-folded normalization. This proves sampled composition, not visual quality or every matrix entry.
- A public-generation API smoke reproduced the simple-scheduler preview inversion (a successful callback aborted after preview 1). With the corrected zero-continue/nonzero-stop convention, the same run delivered two previews and all 22 final frames; cancellation stopped at preview 1 with no final frames. Prepared-model and conditioning cache hits were exercised on the cancellation run.
- The former independent-clock RES orchestration and its derived semantic golden test were removed, rather than renaming ungenerated fixtures to imply verification. The existing real-DiT single-forward fixture test remains; packed-AV sampler behavior is covered by the new independent numerical oracle.
- LCM and RES each produced identical decoded RGB and PCM SHA-256 values across initial generation, same-seed prepared-cache reuse, and cache clearing/rebuild. RNG/history reset and cache equivalence were exercised through actual native generation, not just the synthetic sampler.

## Combined 768p smoke

One requested combined-stack render completed with the actual FL2VA base, Lightx 768p v1.2 (file `3181960`, strength 1) and Unlocked v2.1 (file `3305979`, strength 0.5). The [Unlocked publisher card](https://civitai.com/api/v1/models/2925727) recommends starting at 0.5, CFG 1 and 4–8 steps with Turbo, but specifies Euler ancestral. This run deliberately prioritized Lightx's RES/simple recommendation; it is not claimed to follow both conflicting sampler recommendations.

| Observed property | Result |
|---|---|
| Video | 1344x768, 22 frames, 24 fps; 0.916667 seconds |
| Audio | 32 kHz stereo AAC; container duration 0.925 seconds |
| Sampling | RES multistep / simple, shifts 12/3, 5 observed joint evaluations |
| Execution | 50 blocks, reuse/core reuse 1, native default precision |
| Active adapters | 2; Unlocked 200 matched pairs, Lightx 208 matched pairs |
| Retained adapter snapshots | 2,254,450,970 bytes |
| Process wall time | 106.506 seconds |
| Denoising wall time | 21.495 seconds |
| Peak physical footprint | 24,614,389,192 bytes |
| Maximum RSS | 31,908,675,584 bytes |
| Output SHA-256 | `17020c333b2caa01e98b673f9a364225a0450973364aa7f6c82b072bc3da186b` |

Both audiovisual streams fully decoded. The transferred local MP4 matched the remote hash, and sampled frames showed a fully clothed adult walking and waving in a park. No subjective audio/synchronization verdict is claimed. Existing host-wide swap usage was unchanged before/after; this is not a claim of exclusive machine use or a controlled speed benchmark. The prior ten-second attempt and the full all-files comparison were stopped at the user's request; they are not reported as completed. No additional test-suite runs were performed for this final smoke or documentation cleanup.



