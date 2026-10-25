# Native H3 quality-profile research

Research date: 2026-10-26. Scope: primary-source documentation and existing request records only. No generation, model downloads, builds, tests, deployment, or engine changes were performed. The user's poor high-resolution result is accepted; this note does not try to disprove it. Recommendations below are comparison candidates, not a universal best preset.

## 1. Conclusions and confidence

1. **High confidence:** prompt/reference semantics are an important quality variable, not a cosmetic wrapper. MiniMax explicitly calls Context-IR critical to final quality; its hosted multi-stage system is not released. Following the published prompt format is a reproducible local approximation, **not equivalence to the hosted system**. [S1–S4]
2. **High confidence:** artifact family/version, canvas, shifts and NFE must travel together. Current ModelTC specifications distinguish 544p FL2VA 12/3 from 768p FL2VA 6/3. The newer release's video-shift-6 recommendation explicitly excludes Ref2VA. There is no source here establishing a numeric training shift for Ref2VA 8-step v1.0/v1.1; retain 12/3 as a disclosed local/framework fallback, not a trainer-certified optimum. [S5,S7,L1]
3. **High confidence:** higher resolution alone does not fix semantic adherence, speech, reference leakage, or poor motion. Official 2K output is context-conditioned regeneration, not conventional 2× image/video upscaling. The regeneration module is not open sourced. Generic upscaling is a separate delivery-quality experiment. [S1]
4. **Moderate confidence as an experimental priority, not a quality guarantee:** first compare the existing Ref2VA Turbo8 v1.0 against the same artifact with a properly structured prompt, then an in-range duration; only then compare unaccelerated Ref2VA base and alternative compatible Turbo artifacts. This isolates plausible request-level problems before changing all numerical settings.
5. **Insufficient evidence:** universal superiority of 5–8 versus more steps, “almost no quality loss” for int8/pruning/Turbo, perfect 22/39-frame latent-overlap continuity, or a portable “best” shift/sampler. No controlled benchmark supporting those broad conclusions was found in the inspected primary materials. A released four-step model is not simply an ordinary model run with fewer steps. [S5–S8]

## 2. Official quality pipeline and duration boundary

MiniMax documents H3-Context-IR → H3-Base (768p) → H3-Regenerate-2K. Context-IR parses instructions, cross-modal associations, temporal relationships and logic into structured context; it can supplement underspecified semantics without changing user intent. Both hosted Context-IR internals and the regeneration module are unavailable as local open-source implementations. MiniMax provides API-assisted reproducible workflows and local 768p scripts. The official Ref2VA request script includes all six reference sections and explicit reference/audio relationships, and targets five seconds. [S1,S4]

Published output specifications are **4–15 seconds, 24 fps, default short edge 768, stereo 32 kHz**, with stable support listed for 11 dialogue languages including Chinese. This is an **official supported output range**, not proof that every training example or every Turbo distillation used exactly that duration range. The inspected Turbo model table states training resolution and shifts, not training-duration distribution. Do not relabel the 4–15-second product specification as a measured training distribution. [S1,S5]

At 24 fps, 22 frames are 0.9167 s, 39 are 1.625 s, and the inspected 73-frame request is 3.0417 s. All are below the official four-second lower bound. They can validate wiring/output handling but cannot substantiate full-range motion, complete Mandarin speech, timbre retention or long temporal stability. Below-range duration is a **risk hypothesis**, not an established cause of the observed failure. For a future quality comparison use a supported engine frame count in the 5–7-second range; ModelTC's Comfy guide uses 124 frames (~5.17 s) for a nominal five-second request. Its `17*n+5` rule is a framework contract that should not be imposed blindly on another engine. [S1,S6,L2]

## 3. Ref2VA image, scene and sound fidelity recipe

The official full-reference guide requires six ordered English sections: `subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, `non_diegetic_music`. Dialogue/lyrics retain their original language inside `<d>`; visible scene text also retains its original language. The main detailed description normally spans 350–500 English words for generation, adjusted to preserve a complete spoken timeline rather than padded mechanically. [S2]

- Define **one `<Subject 1>`** for the same person across body/clothing and portrait images; identify what each image contributes. Do not accidentally define two characters because there are two inputs.
- Use a standalone `<Picture N>` role when an image is an actual target frame/composition anchor. An identity-only image should be cited inside the subject definition instead; do not call its studio composition a fully preserved first frame if the intended scene is a new bookstore.
- State what is retained (face, hairstyle, body proportions, clothing) and what changes (environment, composition, pose) within the label's defined role. Official `fully_preserved` is a semantic relationship marker, **not a pixel-equality guarantee**.
- Define `<Audio 1>` as the **voice-timbre reference for `<Subject 1> (S1)`**. `reference` means generated timbre/delivery, whereas `fully_copy` means reusing the complete audio signal; these are different tasks. Do not carry reference words into the new speech when only timbre is requested.
- Put the target speaker event in the shot: `<Subject 1> (S1) ... <d>[Chinese] 你好，今天一起出發吧。</d>`. Speaker identity, delivery and physical action stay outside `<d>`; language and actual spoken words stay inside. Finish speech before the end rather than implicitly demanding cutoff.
- Use one coherent scene and shot timeline. Remove the unused default café suggestion when the actual request is a bookstore; prefer explicit positive spatial/compositional description over a pile of negative directives. The latter is a proposed simplification, not an official guarantee.
- Keep camera, person scale and motion physically compatible. Full-body/shoes-visible framing necessarily gives the face fewer pixels than a close-up; assess identity with that constraint instead of silently changing the requested composition.
- `overall_soundscape` covers room tone and physical sounds; dialogue belongs in the detailed shot. `non_diegetic_music: N/A` explicitly avoids an unwanted score. The base guide reserves soundscape `N/A` for user-requested complete silence. [S2,S3]

For Turbo reference images, ModelTC explicitly recommends **`match`**, matching distillation training: preserve aspect ratio, never upscale a smaller reference, scale down to target pixel area when needed, round dimensions to the 32-pixel grid, avoid cropping. `max` and `diffusers` are different policies (the latter forces short edge 2048 and can upscale), not intrinsically higher-fidelity settings. [S5]

Prompt syntax does not itself implement a first-frame constraint, copy an audio waveform, or prove native encoder/layout/tokenizer compatibility. If exact first-frame conditioning is separately requested, compare actual conditioning mechanisms, not merely the wording “starts from Picture 1.” See the repository's `REF_COMPATIBILITY_RESEARCH.md` for that separate contract. [L1]

## 4. Artifact-specific settings: do not mix source layers

| Artifact | First-party canvas/training evidence | Steps/shifts/sampler evidence | Interpretation |
|---|---|---|---|
| FL2VA Turbo4 v0.1 | 544p mixed aspect | 4 NFE; 12/3; Comfy guide default Euler | Useful lower-resolution control, not a 768p-trained equivalent. |
| FL2VA Turbo8 v1.0 **without** `768p` | 544p mixed aspect | recommended 8 / 4 NFE; 12/3 | Four-step use is explicitly listed for this artifact, not transferable to all Turbo8 files. |
| FL2VA Turbo4 v1.0 `768p` | 1344×768 | 4 NFE; 6/3 | Match this artifact, not the 544p one sharing v1.0. |
| FL2VA Turbo8 v1.0 `768p` | 1344×768 | 8 NFE; 6/3; official Studio README lists same settings | Strong first-party candidate for an eight-step FL2VA quality baseline. |
| Ref2VA Turbo4 v0.1 | 544p mixed aspect | 4 NFE; 12/3 | Wrong substitute for an unmodified Ref2VA768p comparison. |
| FL2VA Turbo4 v1.2 `768p` | release says up to768p | **Euler,4 steps,6/3** in author release #52 | Author's audio-cleanliness claim versusv1.1 has a same-input/prompt/seed/settings video pair. Not a general quality benchmark. |
| FL2VA Turbo4 v1.3 / Turbo8 v1.1 `768p` | release identifies768p files | matching4/8steps; video6 explicitly **FL only** | Newest is not proven best; compatibility regressions are reported in same thread. |
| Ref2VA Turbo4 v1.0 / Turbo8 v1.1 `768p` | release identifies768p files | matching4/8steps; no numeric Ref shift supplied | Exploratory version comparisons, not grounds to change Ref shift to6. |
| Existing native Ref2VA Turbo8 v1.0 `768p` | file identity/current local preset |8 RES/simple,12/3,strength1,match | Reproducible local control;12/3 and RES are not established as trainer-optimal by the inspected ModelTC table. |

Rows derive from [S5–S8,L1]. ModelTC's current README and example graphs still list older artifacts whereas HF discussion #57 announces newer versions: do not use the older table to invent missing newer-version training parameters.

### Official versus local five-step guidance

Current local README/Turbo inventory use FL2VA768p v1.2 at **five steps, RES/simple,12/3**, with12/3 already disclosed as fallback; that is a different recommendation source from original trainer release #52 (**four,Euler,6/3**). Both can be compared, but the local/card recipe must not be presented as the official LightX2V optimum. The latest poor result is **Ref2VA8**, not this FL2VA4 recipe, so correcting FL settings does not by itself diagnose that job. [S8,L1,L2]

### Known fixes and regressions

Release #52 specifically claims cleaner/stabler audio with fewer artifacts in FL2VA4v1.2 versusv1.1 and publishes a controlled illustrative pair. Some commenters dislike the visual detail in the pair; those comments are subjective, not aggregate evaluation. We did not play/download the media or independently score them. [S8]

Release #57 contains reports of degraded new Ref4/8 results and FL4v1.3 problems with additional LoRAs/accelerated attention, with older versions reportedly restoring quality. A participant suggests strength2; the reporting user says stability improves but quality remains poor and other-LoRA combinations still fail. A responding account acknowledges the feedback and says future releases will improve it. This is **not a verified fix**, not grounds for default strength2, and not proof those failures apply to the current native v1.0 job, which used no additional LoRA or approximate attention. [S7,L3]

## 5. Concrete matching: latest inspected remote job

**Candidate matching only:** the user has not identified the exact complained-about output. Parent located the latest inspected job; its request/result were read directly. It must not be called the definitive complained artifact without that identification. [L3,L4]

- Job: `ted:/Users/tedmacbookpro/projects/h3-lora/outputs/video-maker/job-20261025T185809213448Z-dea60a`.
- Ref2VA base directory: `/Users/tedmacbookpro/projects/h3-models/MiniMax-H3-Ref2VA`; a directory path is not a content hash of the base weights.
- Sole adapter: `minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors`, strength1, SHA256 `6a56f41ab4229c9dd845b9501bbd475ee57e112d846cf2e819d534a1ae928c5a`.
-1344×768,73frames,seed2201,8steps,RES multistep/simple,shifts12/3,50layers,reuse/core-reuse1,reference-size match. Result records `H3_VAE_PRECISION=f32`. No second adapter is active despite an unused unlocked-path field in the request profile.
- Actual conditioning order: `front.png`, `portrait.png`, then `voice.wav`. The request's broader input inventory also contains `source.png`, but it is **not** a conditioning argument in the executed command.
- RuntimeSHA256 `0309fff89b1abcdccede9e358fb37d96053c0d975ce779070967dda59f285de7`; elapsed440.067s; output1,146,392bytes, SHA256 `71b3225ac18a0c014a53665e29c284e57d4bb7f2add6f2cb0d102dfd7e8f74ca`.
- Prompt asks for a new full-body bookstore scene and identity/clothing/voice retention, but also includes an unused default café instruction. It uses plain `Picture 1/2`, no `<Subject>` definition/retention sections, no explicit `<Audio 1>` speaker binding and no `<d>[Chinese]` delimiters around `你好，今天一起出發吧。`.

**Plausible risks, not established causes:** below-range3.04s duration; incomplete published Context-IR-style structure; weak role disambiguation across two images and voice; unnecessary fallback setting; very small face within a full-body frame. The manifest does not support blaming low output resolution, a second style adapter, reduced layers/reuse, or VAE low precision. These candidate risks should be isolated experimentally rather than asserted as diagnoses.

Concrete matching additions when available: identify whether the user's disliked file has the recorded output hash; record base-component hashes, source reference dimensions/audio length, exact observed defects/timestamps, and decoder/container frame timing. No new generation is needed to identify these existing artifacts. Do not silently replace the actual job with a prettier showcase request.

## 6. Recommended future comparison design (not executed)

### Phase A: preserve and isolate actual request

1. Keep the existing output/request/runtime identity as **A0**, not regenerate it to dispute the user.
2. **A1 prompt-only:** same exact base/adapter/hash/strength, refs/order,seed2201,1344×768,73frames,8RES/simple12/3,F32VAE; replace only the prompt with faithful six-section Ref2VA structure. Remove fallback café while retaining actual bookstore/full-body/dialogue intent. This deliberately remains short to isolate prompt changes.
3. **A2 duration-only from A1:** same settings/prompt semantic content, but supported5–7s duration, e.g.124frames if accepted by the native request contract. Make speech complete with quiet lead/tail time; do not add extra plot/action. Exact text changes needed to express timeline must be logged.
4. After A1/A2 choose neither by one lucky seed: repeat the principal comparison on fixed seeds2201,2202,2203. Native and PyTorch seeds are not equivalent noise tensors; never label cross-backend same integer seed as pixel-reproducible parity.

### Phase B: checkpoint/sampler comparison at intended quality range

- **Ref control R0:** Ref2VA base, no Turbo, full model/default supported non-Turbo schedule with exact documented parameters. Obtain the existing engine/upstream full-quality NFE recommendation rather than inventing20/30/50 as universal optimum.
- **R1:** current RefTurbo8v1.0,8NFE,RES/simple12/3,match. This is the actual local baseline.
- **R2:** same weights and inputs,8NFE,Euler/simple12/3 to isolate sampler choice; ModelTC's general Comfy default is Euler, but no inspected release certifies this exact Ref8v1.0 combination as optimum.
- **R3 exploratory:** existing RefTurbo8v1.1 if already present and mapped correctly, matching8NFE and disclosed Ref-shift fallback; do not download or silently upgrade. Compare adapter alone first. New version may regress.
- More-step experiment: only after matching artifact's recommended NFE, compare the **same** Turbo8 at12 or16NFEs as an explicitly off-recommendation ablation. Additional NFE is not guaranteed to improve distilled dynamics. Separately compare non-Turbo base at its supported full schedule; these answer different questions.
- FL2VA is a **separate task**: use a genuine first-frame/text-only scenario, FL base and matched FL adapter. Compare FLTurbo8v1.0_768p8steps6/3Euler/simple against FLTurbo4v1.2_768p4steps6/3Euler/simple. Optional local5RES12/3 is a separate recipe; its differences cannot be attributed to step count alone.

Do not stack style/unlocked LoRAs, sparse attention, TeaCache/reuse, layer pruning, lower precision, or upscaling into these initial controls. This is experimental isolation, not a claim all such optimizations degrade quality. Keep resident/cold timing separate if timing is also reported.

### Scoring and acceptance

Parent-only existing-media observation: a six-frame grid of this candidate showed one small central full-body person, a tiny head relative to the canvas, a room dominated by bookshelves/café-like seating, and modest head/pose motion in fixed wide composition. This is not this worker's independent visual assessment, not proof of a defect, and not identification of the user's complained file. It is consistent with the requested full-body/fixed-camera composition. No subjective voice-fidelity assessment was supplied. A **separate framing diagnostic** can compare a mid-shot against the requested full-body shot to measure face-detail sensitivity; it must not replace the full-body acceptance case or be labeled a resolution-only experiment.

Use blinded randomized pairs, identical delivery encoding, full playback with sound, reference-visible identity review, and fixed time-aligned crops. Record separate scores for: facial identity; clothing/body consistency; background transfer versus studio leakage; composition/full-body/shoes; anatomy; temporal flicker/motion; exact Mandarin words and pronunciation; speaker timbre; lip synchronization; audio artifacts; complete sentence/end cutoff. Log catastrophic failures separately instead of averaging them away. Report all fixed-seed outcomes and wall/denoise/encode/decode time when existing instrumentation supplies them.

Predeclare the user's essential criteria (identity, new bookstore, natural full-body motion, correct complete Mandarin and reference timbre) and choose a Pareto candidate; do not let a sharper still compensate for wrong words or an unwanted studio backdrop. No universal numeric pass threshold is inferred from the sources. For cross-framework correctness compare identical injected latents/conditioning and intermediate numerics where possible; output rankings alone cannot localize an implementation discrepancy.

### Resolution/upscaling and continuation arms

Only after semantic/motion quality is acceptable, compare native1344×768 against a **separately specified** low-resolution+2×video-upscaler workflow, with exact source model/resolution/weights, temporal settings and total elapsed time.960×540 is not on ModelTC's multiple-of32 generation grid;960×544 is the documented near-size and has a different aspect ratio. Upscaling to1920×1080 requires a disclosed crop/resize/pad policy rather than claiming exact2× equivalence. A sharp upscaled video does not establish correct recovered identity/words. It is not the official context-aware2K pipeline. [S1,S5,S6]

Official prompts describe video continuation and audio continuity, but that is not evidence of **perfect** continuity for arbitrary22/39-frame overlap recipes. A future continuation comparison must identify actual preserved context/latent indices, frame-time mapping, overlap length, audio alignment, seed/noise policy and join operation. Score boundary identity, pose/background motion, exposure and audio waveform/perceptual discontinuity across several joins, plus drift over the full clip; do not validate only the first overlapping frame. No native continuation implementation or generation was attempted here. [S2]

## Sources

Primary web sources read directly (mutable `main` pages; date above identifies inspection, not an immutable release pin):

- [S1] MiniMax official README: https://github.com/MiniMax-AI/MiniMax-H3/blob/main/README.md
- [S2] MiniMax full-reference prompt guide: https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/ref-en.txt
- [S3] MiniMax base/keyframe prompt guide: https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/base-en.txt
- [S4] MiniMax reproducible768p Ref2VA request (actual script, not showcase title): https://github.com/MiniMax-AI/MiniMax-H3/blob/main/scripts/readme/reproducible-768p-ref2va-request.sh
- [S5] ModelTC model specifications and reference resizing/training policy: https://github.com/ModelTC/Minimax-H3-Turbo/blob/main/README.md
- [S6] ModelTC ComfyUI inference instructions: https://github.com/ModelTC/Minimax-H3-Turbo/blob/main/COMFYUI_SETUP_AND_INFERENCE.md
- [S7] LightX2V owner release and separately attributed feedback thread#57: https://huggingface.co/lightx2v/Minimax-h3-Turbo/discussions/57
- [S8] LightX2V owner release#52, settings and linked same-input/seed audio comparison: https://huggingface.co/lightx2v/Minimax-h3-Turbo/discussions/52
- [S9] Current official HF card (links to model specs, not a standalone complete parameter table): https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/README.md

Local/current observations (not independent evidence of upstream quality):

- [L1] `REF_COMPATIBILITY_RESEARCH.md` and `TURBO_DESIGN.md`: documented native presets, shift fallback, prior smoke scope, sampler contract.
- [L2] `README.md`: current draft/final profiles and native runtime lifecycle. These are local recommendations, not original trainer authority.
- [L3] Remote job `remote-result.json`: `ssh://ted/Users/tedmacbookpro/projects/h3-lora/outputs/video-maker/job-20261025T185809213448Z-dea60a/remote-result.json`.
- [L4] Same job `remote-request.json`: `ssh://ted/Users/tedmacbookpro/projects/h3-lora/outputs/video-maker/job-20261025T185809213448Z-dea60a/remote-request.json`.
