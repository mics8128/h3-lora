# Reproducible Turbo comparisons

`tools/compare_loras.py` is a standard-library Python runner, not a downloader or installer. Run inference only on `ted`. Its public inventory, `tools/turbo_models.json`, records the exact 21 weight filenames, file/version IDs, byte counts, SHA-256 values and file-specific download URLs from the Civitai model API inventory captured on 2026-10-23. No weights are committed.

## Run

On `ted`, with the engine already built and all assets already acquired:

```sh
python3 tools/compare_loras.py \
  --engine-dir . \
  --model-dir /path/to/base-root \
  --ref-model-dir /path/to/reference-base-root \
  --lora-dir /path/to/loras-turbo \
  --reference-image /path/to/park-reference.png \
  --output-dir /path/to/fresh-comparison \
  --suite representatives --execute
```

Both model arguments are **roots containing the original `FL2VA/` and/or `Ref2VA/` trees**, not a transformer directory or an arbitrary checkpoint. The reference root must also include the FL2VA inventory required by engine context construction. Shared weights may be provisioned with verified symlinks outside this runner. The runner never downloads, copies, renames or substitutes model weights.

- `--suite representatives` selects ten representatives plus the separate DMAD LCM discrepancy case: **11 runs**.
- `--suite all` selects all 21 exact artifacts plus that discrepancy case: **22 runs**.
- `--only 3090387,3271482` filters the selected suite by file ID. DMAD file ID selects both Euler and LCM cases; use `--only 3271482-official-lcm` to select only the discrepancy case.
- `--execute` explicitly permits inference. Without it, use `--preflight-only`; the runner does not infer execution permission from the SSH alias or the machine's actual hostname.
- `--preflight-only` checks the selected files, hashes, model inventories and actual executable help without generation. It is permitted on another host, but requires the same macOS tools and provisioned assets. Preflight success is not generation success.
- `--frames` defaults to **124**; output is required to be 24 fps. `--seed` defaults to 42. Shorter clips are smoke tests, not the full quality comparison.
- `--width` and `--height` explicitly override the selected canvas and are recorded as deviations. Defaults are 1344×768, or 960×544 for the non-768 8-step v1.0 and v0.1 families. These are disclosed comparison canvases, not a claim that every card specifies those exact dimensions.
- Use a fresh output directory. Existing summaries are never overwritten by a new invocation. The runner has no resume or fabricated-success mode.

## Settings and qualifications

Every case uses one Turbo adapter alone, native CFG 1, `simple`, 50 blocks, denoiser reuse 1 and core reuse 1. Token reduction, internal reduced canvas and SSD streaming are not requested. Default native precision is retained; the exact binary hash, git revision/worktree status, hardware and engine diagnostics accompany the results. This is not a controlled speed benchmark: sampler, steps, resolutions and reference conditioning differ.

| Family | Representative file | Steps | Sampler | Strength | Video/audio shift |
|---|---:|---:|---|---:|---|
| Larry v4 EMA | 3090387 | 6 | res_multistep | 1 | 12/3 |
| DMAD–Larry full merge | 3281488 | 6 | lcm | 1 | 12/3 |
| DMAD full | 3271482 | 6 | euler | 1 | 12/2 |
| Taomate original EMA conversion | 3208044 | 4 | euler | .75 | 12/3 |
| Lightx 768 v1.2 | 3181960 | 5 | res_multistep | 1 | 12/3 |
| Lightx Ref2VA | 3179683 | 8 | res_multistep | 1 | 12/3 |
| Lightx 768 v1.0 | 3101219 | 4 | euler | 1 | 6/3 |
| Lightx 8-step v1.0 | 3101211 | 4 | euler | 1 | 12/3 |
| Lightx v0.1 | 3088145 | 4 | euler | 1 | 12/3 |
| Lightx 768 v1.1 | 3159814 | 5 | res_multistep | 1 | 12/3 |

Larry non-EMA step600 also uses six-step RES; older Larry files use **experimental four-step RES**, inheriting strength/shifts without an independently established best-card recommendation. Compressed alternatives inherit their family settings and are not claimed to have independent best settings. Larry's pruned artifacts cannot restore removed AdaLN updates.

Sources:

- [Civitai model inventory/card](https://civitai.com/api/v1/models/2837571), captured 2026-10-23, supplies the current family recommendations and exact file identities. This API URL is mutable; the committed manifest preserves the dated selected values and weight hashes, not a claim of an immutable card revision.
- [Original Lightx specifications, pinned revision](https://github.com/ModelTC/Minimax-H3-Turbo/blob/02e26d591f7a04d5d1a074c9566d5dd4f22f6225/README.md) and [Comfy inference instructions](https://github.com/ModelTC/Minimax-H3-Turbo/blob/02e26d591f7a04d5d1a074c9566d5dd4f22f6225/COMFYUI_SETUP_AND_INFERENCE.md) supply the disclosed fallback for the older versions. The non-768 v1.0 recommendation says **“8 / 4”**; four is the selected minimum. The 768 v1.0 uses **6/3**, not the 12/3 default.
- [Original DMAD instructions, pinned revision](https://github.com/Yzmblog/DMAD/blob/32ac967324eb0ebba654faeffda99f5def1e86cb/comfyui/README.md) prescribe LCM/simple. The extra `3271482-official-lcm` run retains six steps to isolate the sampler discrepancy against the Civitai six-step Euler run. It does **not** claim six is the original trained setting (four).
- [Pinned Comfy ModelSamplingAV](https://github.com/Comfy-Org/ComfyUI/blob/bc5f880dbdd43e426077f42ccec53bb577ffee9f/comfy/model_sampling.py) documents the packed audiovisual semantics; 12/3 where omitted by cards is an explicitly adopted default, not a per-version optimum.

Ref2VA uses the real reference base, ordered `<Picture 1>` conditioning, `match` sizing, and a prompt explicitly requesting that image's adult identity/clothing. It is a separate conditioned task, not a directly interchangeable T2VA ranking row. The generic benign prompt is authored for this comparison. Native seeded RNG is not a promise of PyTorch-identical noise.

## Evidence and failure behavior

Before any inference, selected adapter bytes and full SHA-256 values must match; every base index/config is recorded and every indexed safetensors shard must exist. Base shards are inventoried by size, **not falsely described as cryptographically authenticated**. The executable must expose all used flags. Engine binary hash, git commit and uncommitted status identify the implementation actually invoked.

Each case directory contains:

- `result.json`: exact argv and shell rendering, settings/card source, prompt and hash, reference hash where applicable, requested parameters, observed native sigma arrays and NFE, timing, exit status, actual ffprobe media metadata, and successful output SHA-256.
- `engine.log`: full engine loading/sampling diagnostics and macOS `/usr/bin/time -l` output, including maximum RSS and peak physical memory footprint. Preserve this log to inspect adapter coverage and effective precision. Under `--profile` the native sampler emits its actual resolved video/audio sigma arrays and counted joint model evaluations; the runner requires one successful trace matching the requested sampler, grid length and NFE. It does not substitute a theoretical grid for observed values.
- `decode.log`: complete audio/video `ffmpeg -xerror` decoding evidence.
- `output.mp4`: only considered successful after generation exits zero, ffprobe confirms video **and audio**, requested dimensions/frame count/24 fps match, and full audiovisual decode succeeds.

`summary.json` is updated after every completed case. Case failures do not prevent later cases from running. Any failed case or preflight returns nonzero; no placeholder, silent settings fallback or replacement artifact is emitted. Swap and VM snapshots are before/after host-wide observations, not per-process swap attribution. Decode success establishes media integrity, not subjective quality, reference fidelity, motion quality or audiovisual synchronization. Those require viewing/listening and separate reporting. Transfer final media to a dedicated local Movies directory and compare local SHA-256 with each remote result before playback; the runner does not transfer weights or publish results.

## Licenses

The source code license does not grant rights to the H3 base or any LoRA. Obtain and inspect the base-model license, each original trainer's terms and each conversion card's usage restrictions separately before use or redistribution. The inventory is a reproducibility reference, not a redistribution permission or a universal statement that all artifacts share a license. Reference-image permission is independently required.
