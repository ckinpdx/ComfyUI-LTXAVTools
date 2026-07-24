# ComfyUI-LTXAVTools

Utility nodes for LTX-2.3 audio-video generation and LoRA training workflows in ComfyUI.

---

## Nodes

### LTX Dimension Calculator
Aspect-ratio-aware resolution picker. Outputs only LTX-compatible resolutions (divisible by 64). Dynamic dropdown updates when ratio or orientation changes.

**Outputs:** `width`, `height`, `width_half`, `height_half`, `label`

**Custom override** (optional): toggle `use_custom` to override the dropdown with `custom_width`/`custom_height` — for input videos at resolutions the dropdown doesn't list. The toggle (not a value check) controls the mode, so bypassing an upstream node feeding the custom dims won't accidentally switch it (and if `use_custom` is on but the dims arrive ≤ 0, it warns and falls back to the dropdown).

`custom_role` sets how the custom size maps to the two stages:
- **`half (stage 1)`** — the custom size is your stage-1 resolution; the node snaps it to ÷32 and sets `width`/`height` = **2× custom**. Use this for continuing/encoding an input video: stage 1 runs at the source resolution and stage 2 doubles. (e.g. custom 960×512 → half 960×512, full 1920×1024.)
- **`full (final)`** — the custom size is the final resolution; snapped to ÷64 so the derived half stays ÷32. (custom 960×512 → full 960×512, half 480×256.)

### LTX Dimension Calculator 3 Stage
Same as above but constrained to div-by-128 resolutions, compatible with 3-stage pipelines (full / half / quarter resolutions all remain div-by-32). Matching the base calculator, a `use_custom` toggle overrides the dropdown with `custom_width`/`custom_height` interpreted by `custom_role` — `quarter (stage 1)` scales ×4 to full (the input-video case: stage 1 matches the source), `half (stage 2)` ×2, `full (final)` derives the stages down. Each role snaps to its own grid (÷32 / ÷64 / ÷128) so every derived stage stays LTX-valid; `label` shows the full `quarter → half → full` chain.

**Outputs:** `width`, `height`, `width_half`, `height_half`, `width_quarter`, `height_quarter`, `label`

### LTX Frame Calculator
Snaps a desired duration to the nearest valid LTX frame count — `(frames - 1) % 8 == 0` — and returns the actual snapped duration. Optionally appends extra latent frames after snapping (e.g. a contamination buffer frame) without disturbing the snap.

| Input | Description |
|---|---|
| `seconds` | Desired clip length |
| `fps` | Frames per second |
| `extra_latent_frames` | Extra latent frames added after snapping (default 0) |

**Outputs:** `frame_count`, `latent_frames`, `actual_seconds`, `clean_frame_count`, `clean_latent_frames`

`clean_frame_count` and `clean_latent_frames` exclude the extra frames — use these as trim targets when stripping a contamination buffer.

### LTX Scene Length Calculator
Authors variable scene lengths for the AV Looping Sampler. Takes scene durations in seconds (`10 | 4.5 | 12.2`), snaps each to latent granularity (multiples of 8 pixel frames), and outputs the `scene_lengths` string for the sampler **and** the exactly-matching total `frame_count` for the empty latent — single source of truth, so schedule and canvas cannot disagree.

| Input | Description |
|---|---|
| `scene_seconds` | Scene durations in seconds, `\|` or `,` separated — one per chunk/prompt segment |
| `fps` | Frames per second (25 for AV) |

**Outputs:** `scene_lengths`, `frame_count`, `scene_count`, `info`, `actual_seconds`

`actual_seconds` (= `frame_count / fps`) matches the LTX Frame Calculator convention — use it to trim input audio to the exact video duration in conditioned-audio workflows.

### LTX Video Cut Marker (Scenes)
Interactive timeline widget for authoring a scene schedule against real media. Load a **video or audio file** (picker, upload button, or drag-and-drop) — audio files show a **waveform** on the timeline (videos show their soundtrack's waveform too), so scenes can be cut against a song's sections visually. Cuts mark scene boundaries, hard-snapped to the LTX latent grid; an optional **end marker** (red, `E`) truncates the schedule, and an optional **start marker** (blue, `S`) trims the head — the schedule is then measured across the `[start, end]` window.

Indices are **time-anchored and emitted in `emit_fps` frame space** (default 25): a 24fps source marked while emitting at 25 — the VHS `force_rate` case — produces a schedule in the *pipeline's* frame space, not the file's.

An **auto-cut** control (interval box + ⚡ button) replaces all cuts with one every N seconds, grid-snapped, from the start marker to the end. Loading **new media** resets the schedule (the workflow-restore path is untouched, so a page refresh still restores saved state).

Controls: double-click / `C` = add cut · drag = move (snapped) · click = select · ←/→ = nudge one latent · right-click / `X` = delete · `S` = start marker · `E` = end marker · Space = play · Alt+←/→ = fine playhead step. (`Del` is intentionally *not* a marker-delete — it stays ComfyUI's delete-node.)

| Output | Description |
|---|---|
| `scene_lengths` | Pipe-separated px counts incl. the final scene → sampler `scene_lengths` |
| `frame_count` | `sum − 7` — the matching empty-latent length (of the `[start, end]` window) |
| `video_path` | → VHS Load Video (Path); this node does not decode |
| `frame_load_cap` | = `frame_count` — cap the upstream loader to exactly the scheduled frames |
| `skip_first_frames` | Start marker's pixel-frame offset → VHS loader's `skip_first_frames` (0 = from the start) |

### LTX Keyframe Planner
Turns a `scene_lengths` schedule into **end-anchored keyframe indices** for keyframe-travel generation: frame `0` opens (optional), each scene travels to a keyframe at **its own end** — landing in its own chunk so generation converges on the image and the next scene inherits through the overlap carry — and the final scene closes on `-1` (optional). `128|128|96` → `0,120,248,-1`.

**Outputs:** `indices` (→ `optional_cond_image_indices`), `count` (must equal the images stacked into `optional_cond_images`, in order), `info` (per-image timestamps receipt).

### LTX Keyframe Pair Concat
Emits consecutive keyframe pairs as one composite image for **vision-LLM scene prompting**: index 1 → images 1+2, index 2 → 2+3… Under the end-anchored plan, **pair k is exactly scene k's travel endpoints**, so a VLM shown the pair writes scene k's transition prompt. Drive `index` with an incrementing INT primitive across queue cycles; `total_pairs` (= batch − 1) is the cycle bound.

| Input | Default | Description |
|---|---|---|
| `index` | 1 | 1-based pair index (clamped to last valid pair) |
| `direction` | horizontal | earlier keyframe left/top, later right/bottom — use **vertical** for landscape keyframes (near-square composite preserves panel detail through vision-encoder resizing) |
| `gap` | 8 | Black divider px between panels — helps the VLM read two distinct panels |

**Outputs:** `image`, `pair_info` (e.g. `pair 2/3: keyframe 2 → 3`), `total_pairs`.

### LTX LoRA Metadata Reader
Reads a LoRA's safetensors metadata header — no weight loading, no VRAM, milliseconds. The **single point of LoRA selection** for IC-LoRA workflows: one combo drives both the loader and the sampler, so the factor can never drift from the loaded file.

```
Metadata Reader ── lora_path ─────────────▶ KJ LTX2 LoRA Loader Advanced .opt_lora_path
              ├── latent_downscale_factor ─▶ AV Looping Sampler .guiding_downscale_factor
              └── metadata ────────────────▶ (inspection — works on any LoRA's header)
```

The factor comes from the LoRA's own `reference_downscale_factor` metadata (pixel spatial upscaler x2 = 2, x4 = 4). When `opt_lora_path` is connected, the KJ loader's own combo is ignored.

### LTX AV Streaming Decode & Save
Long-video export without the RAM cliff: decodes the video latent in temporal chunks and pipes frames directly into a persistent ffmpeg encode — **the full pixel tensor never exists**, so RAM use is constant at any length. Each chunk decodes with `context_latents` of leading context that are trimmed from the output; because the LTX video VAE is temporally causal this makes the chunked decode **exact** (bit-identical to a full decode, no blending, no seams). Frame count preserved exactly (`(T−1)×8+1`). Shows an inline player when finished (streams from disk — previewing also costs no RAM).

> **When to use:** it is **slower than the normal decode path** — reserve it for videos long enough that VAEDecode → VHS would exhaust RAM. For ordinary lengths, use the standard pipeline.

| Input | Default | Description |
|---|---|---|
| `latent` | — | AV NestedTensor (video component used) or plain 5D video latent |
| `chunk_latents` | 16 | Latent frames decoded per chunk (~chunk×8 pixel frames of RAM at a time) |
| `context_latents` | 4 | Causal left-context per chunk, trimmed from output. Raise if a seam ever appears at chunk boundaries |
| `fps` / `crf` | 25 / 19 | Encode parameters (libx264, yuv420p) |
| `optional_audio` | — | Decoded AUDIO (via `LTXVAudioVAEDecode` — audio is tiny, decode it normally) muxed into the file |

**Output:** `file_path` (STRING). Note there is deliberately no IMAGE output — pixels stream out and are gone; anything needing frames works from the latent or the saved file.

### LTX Streaming Video Encode
The input-side mirror: encodes a video **file** to latents in temporal chunks — frames are read from disk one chunk at a time, so the full pixel tensor never exists and RAM is constant at any source length. Each chunk is encoded with `context_latents` of left pixel context (plus the 1-frame head) which are trimmed from the output; causal-exact, same trick as the decode side in reverse. Long-source v2v guides (pixel upscale, clean plate, control videos saved to disk) without the IMAGE-batch RAM cliff.

| Input | Default | Description |
|---|---|---|
| `video_path` | — | Source file — wire from Video Cut Marker's `video_path` or type a path |
| `width` / `height` | 0 | Resize before encode (0 = native), snapped ÷32. For small-grid IC guides set gen/factor here |
| `chunk_latents` | 16 | Latent frames encoded per chunk (~chunk×8 pixel frames of RAM at a time) |
| `context_latents` | 4 | Causal left-context per chunk, trimmed. Raise if an A/B vs full encode ever shows a boundary diff |
| `force_rate` | 0 | Resample fps while reading (0 = native). **Set this to your emit/consume rate (25), the same as the VHS loader** — the resampler duplicates (upsample) or drops (downsample) frames so its output-frame indices match VHS's stream, which is what lets the Cut Marker's `skip_first_frames` / `frame_load_cap` land on the same content in both. Mismatched rates offset the encode start. |
| `frame_load_cap` / `skip_first_frames` | 0 | Read window (cap 0 = all) |

**Outputs:** `latent`, `num_latents`, `num_frames`. Sources ending mid-latent are trimmed to the valid `(T−1)×8+1` count with a console note. Preprocessed branches (DWPose/depth) save their video to disk first, then stream-encode the file.

---

### LTX Inpaint Color Fill
Composites a solid fill color where the mask is active — inpaint-guide prep for IC-LoRAs that read the mask from the reference **pixels** (unlike `optional_denoise_mask`, which is the LoRA-less base-model path). Color is a preset (magenta `255,0,255` / chroma green `0,255,0` / Lightricks green `102,255,0`) or a custom hex, so it covers every LoRA's trained convention — core's `LTXVInpaintPreprocess` hardcodes one green. `binarize` (default on) thresholds the mask so the fill color is exact even from soft/grown masks. **Composite at the final encode resolution** (resize source and mask first) — resizing after compositing smears the boundary into off-colors the LoRA never trained on.

| Input | Default | Description |
|---|---|---|
| `images` | — | Video frames |
| `mask` | — | White = filled with the color; single-frame masks broadcast to the video length |
| `color` | magenta | Preset or `custom` |
| `custom_hex` | `#FF00FF` | Used when `color = custom` |
| `binarize` | true | Threshold the mask at 0.5 for an exact fill |

**Output:** `image`. Only needed for the IC-LoRA inpaint route; base-model inpainting uses `optional_denoise_mask` alone (no color fill).

### LTX Video Outpaint Latent
Latent-space **outpaint** prep for the base-model path (no LoRA, no fill color). Zero-pads an encoded video latent — real content in the centre, **zeros in the margin** — and emits the matching feathered denoise mask. The zero margin is *why it works where padding pixels fails*: encoding padded pixels (black / grey / green) bakes **structured content** the model reads as "stuff is here" and tries to preserve, whereas a zero margin is an empty generation target — the same substrate a from-scratch generation starts from. Run in a **full-denoise pass** so the sampler noises and regenerates the margin (a low-denoise refinement won't add enough noise to a bare margin).

| Input | Default | Description |
|---|---|---|
| `samples` | — | Encoded **video** latent (5D `[B,C,T,H,W]`) — not an AV NestedTensor. Separate video, outpaint, re-concat audio |
| `left` / `top` / `right` / `bottom` | 0 | Per-edge padding in pixels, snapped to the LTX spatial grid (÷32) |
| `overlap` | 0 | Full-regenerate band (px) of the original at the seam — rewritten so the seam sits inside one continuous generation, not on a partial-keep line. Keep small (~16–32) |
| `feather` | 32 | Mask ramp (px) beyond the overlap, regen→keep, into the original |
| `margin_fill` | zeros | `zeros` (correct — the sampler noises it) or `noise` (pre-populate for low-denoise passes, experimental) |

**Outputs:** `latent` (zero-padded video), `denoise_mask` (feathered). Wire `latent` (after `LTXVConcatAVLatent` with your audio) → sampler `latents`, and `denoise_mask` → `optional_denoise_mask`. Prompt describes the extended scene (no LoRA prior). Fixes the black-artifact failure, not the one-sided-context limit — strongest on moving-camera / simple margins; large static extensions are where the outpaint IC-LoRA (Inpaint Color Fill + guide) still has a role.

### LTX Audio Latent Trim
Trims a 4D audio latent `[B, C, T, F]` along the temporal axis. Supports negative indexing. Used to extract context and output windows in sliding-window loops.

| Input | Description |
|---|---|
| `audio_latent` | 4D audio latent |
| `start_index` | Start frame (negative = from end) |
| `end_index` | End frame inclusive (-1 = last) |
| `strip_mask` | Remove noise_mask after trim |

**Output:** trimmed `LATENT`

### Latent Strip Mask
Removes `noise_mask` from any latent dict. Useful before feeding latents into `LTXVAddLatents` to prevent mask merge errors.

### LTX Audio Only Latent
Creates an AV NestedTensor for **audio-only generation**: a zero audio latent of the requested duration paired with a minimal 1-frame dummy video latent. Wire the output into `LTXVAudioVideoMask` (KJNodes) using `audio_latent_frames` to set up the denoise mask before sampling.

Not a valid input for the AV Looping Sampler — the dummy video component would define a 1-frame output.

| Input | Description |
|---|---|
| `audio_vae` | Audio VAE (provides channels, frequency bins, latents/second) |
| `seconds` | Audio duration |
| `batch_size` | Batch size |

**Outputs:** `latent` (AV NestedTensor), `audio_latent_frames`

### LTX AV Extend Latent
Prepares an AV NestedTensor for extending existing content. Appends zero latents (video plus time-matched audio) for the extension region and pre-builds the noise mask: `existing_denoise` over the existing frames (0.0 = fully preserve), 1.0 over the new region.

Wire `extension_start_frame` into `optional_cond_image_indices` on the looping sampler and supply the last frame of the existing video as the keyframe image, giving the model a hard visual anchor at the transition.

| Input | Default | Description |
|---|---|---|
| `av_latent` | — | Existing AV NestedTensor (encoded content) |
| `vae` | — | Video VAE (supplies the time scale factor) |
| `extension_seconds` | 3.0 | Duration to add beyond the existing content |
| `fps` | 25.0 | Must match the AV latent |
| `existing_denoise` | 0.0 | Mask value over existing frames; 0.1–0.3 allows light refinement near the transition |

**Outputs:** `av_latent`, `extension_start_frame`, `last_existing_frame` (pixel-frame indices)

### LTX Audio Latent Pad
Pads a 4D audio latent `[B, C, T, F]` by repeating its last frame N times (default 7). Closes the 7-frame audio gap that appears at concatenation boundaries in manual sliding-window loops due to LTX's first-frame asymmetry. Strips any noise mask.

---

### LTX Add Audio Latent Guide

> **⚠️ ARTIFACT (2026-07-15) — retested, no effect without the ID-LoRA. Kept as a record; do not build on it.** It sets `ref_audio` tokens, which the AV model prepends at negative temporal coords with timestep 0 (`av_model.py` `_process_input`) — but that placement is an ID-LoRA training convention, and the base model is deaf to it. The pathway only functions with the TalkVid ID-LoRA (use the core `LTXVReferenceAudio` node for that). The working base-model voice-reference mechanism is **carry-swap** — see [SPEC_NEG_REF_AUDIO.md](SPEC_NEG_REF_AUDIO.md).

Injects a raw audio latent as `ref_audio` conditioning tokens. Input must be a raw 4D audio latent `[B, C, T, F]`. Use `LTXVSeparateAVLatent` first if you have a combined AV latent.

| Input | Description |
|---|---|
| `positive` | Positive conditioning |
| `negative` | Negative conditioning |
| `audio_guide_latent` | 4D audio latent to inject as reference |

**Outputs:** `positive`, `negative`

### LTX Crop Audio Guide
Removes `ref_audio` from conditioning after sampling. Pair with LTX Add Audio Latent Guide — call after the sampler so a stale audio reference doesn't leak into subsequent passes. No latent trimming is needed (audio guides don't append tokens to the latent).

**Inputs:** `positive`, `negative` — **Outputs:** `positive`, `negative`

### LTX AV Reference Audio Multi (ID-LoRA)

> **⚠️ ABANDONED (2026-07-15) — multi-voice does not work.** The ID-LoRA's `ref_audio` is attended *globally* by every audio frame and the identity-guidance pass amplifies it *globally*, so it is **inherently single-voice**; per-chunk reference switching cannot localize a single global voice, and multi-speaker runs degrade (garbled onsets, wrong-speaker bleed). The node is kept as a record and remains a valid **single-reference** drop-in for the core `LTXVReferenceAudio` (speaker 1 only). For real multi-voice, see [SPEC_NEG_REF_AUDIO.md](SPEC_NEG_REF_AUDIO.md) (negative-index reference audio — specced, not built) and the ABANDONED header in `nodes/speaker_ref.py` for the full reasoning.

Multi-speaker version of the core `LTXVReferenceAudio` node. Encodes up to four reference voices (~5 s clean clips). Speaker 1 is applied as the default `ref_audio` and the identity-guidance model patch is identical to the core node — with a single reference this is a drop-in replacement. All encoded voices are attached to the positive conditioning as a `ref_audio_bank` (the multi-voice path the sampler *would* read — non-functional per above).

| Input | Description |
|---|---|
| `model`, `positive`, `negative` | passed through (model cloned + patched) |
| `audio_vae` | LTXV audio VAE |
| `reference_audio_1` | Speaker 1 — default voice (required) |
| `reference_audio_2..4` | Additional speakers (optional — non-functional, see above) |
| `identity_guidance_scale` | Extra no-reference pass per step amplifies speaker identity; 0 disables |
| `start_percent` / `end_percent` | Sigma range where identity guidance is active |

### LTX AV Speaker Prompt Provider

> **⚠️ ABANDONED (2026-07-15)** — the per-chunk `[SPEAKER n]` routing half of the abandoned multi-voice ID-LoRA approach above. Kept as a record; not a working multi-voice path.

`MultiPromptProvider` with voice routing for multi-speaker ID-LoRA generation. Splits prompts on `|` (one per temporal chunk). Segments using `[SPEAKER n]:` in place of `[SPEECH]:` select voice *n* from the reference bank for that chunk; the tag is rewritten to `[SPEECH]:` **before** encoding, so the model only ever sees its trained format. Untagged segments use the default voice. One speaker per chunk — turn-based dialog only.

**Inputs:** `prompts`, `clip` — **Output:** `conditionings` (wire to `optional_positive_conditionings`)

---

### LTX AV Cross-Attention Toggle
Switches the LTX2.3 AV model's cross-modal attention couplings on/off via `transformer_options` (read at `av_model.py` `run_a2v` / `run_v2a`). Takes a `MODEL`, returns a patched clone (source untouched). Both default **on**, so it is a no-op until you flip one. Works with any sampler — wire it on the model line before the sampler.

| Input | Default | Description |
|---|---|---|
| `v2a_cross_attn` | True | Video→audio coupling. Turn **off** when *generating* new-word audio over a guide video whose lips articulate different words — otherwise the guide's visible lips out-muscle the text (lips↔audio is near-deterministic in training, text→audio is one-to-many) and corrupt generated speech into gibberish. Off = audio driven purely by the text prompt; a2v still syncs the video mouth to the new audio. No effect when audio is supplied as input (frozen) — v2a only bites during audio *generation*. |
| `a2v_cross_attn` | True | Audio→video coupling — this **is** lipsync (the mouth following the audio). Turn off only to fully decouple video from audio; disabling it kills lipsync. Rarely wanted. |

**Use case — changing the words over a guide video:** the guide's lips say the original words; asking the audio branch to generate *different* words while those lips are visible produces gibberish. Set `v2a_cross_attn = False` and the text drives the audio cleanly while a2v resyncs the (LoRA-freed) mouth to the new speech.

---

### LTX Distilled Sigmas
Generates a sigma schedule for **distilled or heavily distill-LoRA weighted LTX models** using a high-cluster + cliff + power-tail structure. Not suitable for full dev model runs. Concentrates steps near sigma=1.0, then uses a single large cliff step exploiting the distilled model's learned shortcut, followed by a power-curve tail to zero.

At 8 steps with defaults approximates the known community schedule: `1.0 → ~0.975 (linear cluster) → 0.65 (cliff) → ~0.0 (power tail)`.

| Input | Default | Description |
|---|---|---|
| `steps` | 8 | Total denoising steps |
| `high_fraction` | 0.5 | Fraction of steps in high-sigma cluster |
| `cluster_width` | 0.025 | Sigma range of cluster (1.0 down to 1.0 − width) |
| `cliff_sigma` | 0.65 | Landing sigma after the cliff step |
| `tail_power` | 2.0 | Power curve exponent for tail (higher = more steps near cliff) |

**Output:** `SIGMAS`

---

### LTX Sigma Resample
Resamples a sigma schedule to a different step count while preserving its essential character — cliff positions, cluster density, tail shape. Extracts the f(σ) phase portrait from the source schedule and integrates forward at the new step count. Chain from the same sigma source with different `steps` inputs to produce multiple step-count variants.

| Input | Description |
|---|---|
| `sigmas` | Source sigma schedule |
| `steps` | Number of steps for the output schedule |

**Output:** `SIGMAS`

---

## Utility Nodes

### LTX AV Latent Upsampler
AV-aware wrapper around the LTX latent upscale model. Upsamples the video component of an AV NestedTensor (or a plain video latent); audio passes through unchanged. Processes the full tensor in one shot — the upscaler's GroupNorm normalizes across T×H×W jointly, so temporal chunking shifts the statistics and causes seams regardless of overlap. Tries GPU first and falls back to CPU on OOM.

### LTX AV Latent Upsampler (Tiled)
Temporally tiled variant: overlapping temporal tiles are upsampled on GPU and blended back with a linear crossfade. Viable when the result feeds a low-sigma refinement pass, which smooths residual tile-statistics differences. Use the non-tiled version when exact full-tensor statistics matter.

**Supports both spatial and temporal upscalers — auto-detected** from the first tile's output shape (`L → L` spatial; `L → 2L−1` temporal, per the first-frame asymmetry: the LTX temporal upsampler doubles the pixel timeline, so `T` latents become `2T−1`). In temporal mode, tiles are anchored at `2×` their input position and each non-first tile's head latents are dropped (`head_trim`) — tile heads are malformed video-start latents; the previous tile owns that region and the crossfade spans the remaining `2·overlap−1−head_trim` latents. See [SPEC_TILED_TEMPORAL.md](SPEC_TILED_TEMPORAL.md). Temporal output is 50 fps material — single-shot refinement only (the AV Looping Sampler's audio math is 25 fps).

| Input | Default | Description |
|---|---|---|
| `tile_frames` | 16 | Latent frames per temporal tile |
| `tile_overlap` | 4 | Latent frames of blend overlap |
| `head_trim` | 2 | Temporal mode only (ignored in spatial): output latents dropped from each non-first tile's head. Raise (with `tile_overlap`) if tile joins show motion stutter |

### LTX AV Latent Check
Verifies that the video and audio components of a combined AV NestedTensor are time-matched. Reports video latent frames, audio latent frames, expected audio frames (`(T_v − 1) × 8 + 1` pixel frames at 25 audio latents/second), the delta, and an `is_matched` boolean. Passes the latent through unchanged.

### LTX AV Separate Check
Same alignment math for split video + audio latents. Place after trim operations to verify the pair is still in sync.

### Preview Image Passthrough
Displays an image preview and passes the image through unchanged. Useful inside loops where terminal PreviewImage nodes don't refresh per iteration.

---

## LTX AV Looping Sampler

Temporal (and optionally spatial) tiling sampler for long-form video+audio generation with the LTX2.3 AV model. Generates video and audio jointly as a NestedTensor latent across multiple overlapping chunks, accumulating a coherent sequence longer than any single context window.

**See [LOOPING_SAMPLER_GUIDE.md](LOOPING_SAMPLER_GUIDE.md)** for the field guide: how the input levers interact across modalities, generation regimes, recipes, and a symptom→lever troubleshooting table.

Input latent must be an AV NestedTensor sized to the full output — the video component defines resolution and frame count, the audio component the matching audio length. Build it with the core `LTXVConcatAVLatent` node (video latent + audio latent), an AV VAE encode, or `LTX AV Extend Latent`. Do **not** use `LTX Audio Only Latent` here — its video component is a 1-frame dummy intended for audio-only generation. Use `LTXVLoopingSampler` (ComfyUI-LTXVideo) for video-only generation.

**Output:** `denoised_output` — AV NestedTensor LATENT

### Required inputs

| Input | Default | Description |
|---|---|---|
| `model` | — | LTX AV model |
| `vae` | — | VAE |
| `noise` | — | Noise (from RandomNoise or similar) |
| `sampler` | — | Sampler |
| `sigmas` | — | Sigma schedule |
| `guider` | — | Guider (e.g. CFGGuider) |
| `latents` | — | AV NestedTensor latent defining the generation shape |
| `temporal_tile_size` | 80 | Pixel frames per temporal chunk. Multiple of 8, minimum 24. |
| `temporal_overlap` | 24 | Pixel frames of overlap between chunks. The overlapping region from the previous chunk is injected as a guide so the model maintains visual continuity. |
| `guiding_strength` | 1.0 | Conditioning strength for guiding latents (IC-LoRA / latent guides). |
| `temporal_overlap_cond_strength` | 1.0 | Noise mask strength for the video carry-over (overlap) region. Keep at 1.0 for AV generation — frozen symmetric context (video and audio both 1.0) is what holds lipsync across chunks. |
| `audio_overlap_cond_strength` | 1.0 | Noise mask strength for the audio carry-over region. Keep at 1.0, matching the video overlap — asymmetric anchors cause per-chunk lipsync dropout. |
| `cond_image_strength` | 1.0 | Noise mask strength for image keyframe conditioning. |
| `horizontal_tiles` | 1 | Number of spatial tiles horizontally. |
| `vertical_tiles` | 1 | Number of spatial tiles vertically. Audio is accumulated from tile (0,0) only. |
| `spatial_overlap` | 1 | Latent-space pixels of spatial overlap between tiles. |
| `video_fps` | 25.0 | Must match the fps of the AV latent. Used for audio frame alignment. LTX2.3 AV is trained at 25 fps. |

### Optional inputs

| Input | Default | Description |
|---|---|---|
| `optional_cond_images` | — | Image(s) to condition on. Paired with `optional_cond_image_indices`. |
| `optional_cond_image_indices` | `"0"` | Comma-separated pixel-frame indices for each conditioning image. Negative indices count from the end (e.g. `"0,-1"` for first and last frame). Index 0 is treated as I2V (encoded directly into the latent); other indices are added as guide tokens. |
| `optional_guiding_latents` | — | Pre-encoded video latent for IC-LoRA guided generation. Sliced to each chunk window automatically. |
| `optional_negative_index_latents` | — | Latent added as a guide at the last temporal position, used to steer the model away from a specific visual state. |
| `optional_positive_conditionings` | — | List of conditionings cycled per chunk (one per temporal tile). Allows per-chunk prompt variation. |
| `optional_normalizing_latents` | — | Reference latent for AdaIN normalization. When provided, each chunk's channel statistics are normalized per-frame to match this reference. |
| `adain_factor` | 0.0 | Strength of AdaIN (Adaptive Instance Normalization) applied to each chunk's output. Counteracts tonal drift and overbaking of static regions in long sequences. `0.0` = off. With `optional_normalizing_latents`: per-frame per-channel normalization. Without: uses the first chunk's output as a global channel reference. |
| `audio_cond_strength` | 0.0 | Strength of conditioning on the input audio latent. `0.0` = ignore input audio, generate freely. `1.0` = fully hold the input audio. Intermediate values allow guided regeneration. |
| `scene_lengths` | `""` | Optional pipe/comma-separated pixel-frame counts (multiples of 8), one per chunk, for variable scene lengths. Empty = uniform `temporal_tile_size` chunks. Prompts map 1:1 to scenes. Pair with LTX Scene Length Calculator. |
| `per_tile_seed_offsets` | `"0"` | Comma-separated per-chunk seed offsets. A nonzero entry re-rolls only that chunk's noise (`0,0,7` re-rolls chunk 2) — surgical fix for one bad chunk. |
| `optional_phase2_sampler` | — | Second-phase sampler for dual-sampler schedules. Takes over at `phase2_start_step` within every chunk, resample-style continuation (Clownshark-chain pattern). |
| `optional_phase2_guider` | — | Guider for phase 2 (e.g. CFG 1.0 vs. phase 1's 2.0). Its conditioning is replaced by the chunk's; only guidance settings apply. Falls back to the main guider. |
| `phase2_start_step` | 0 | Schedule step where phase 2 takes over. 0 = disabled. |
| `optional_prior_av_latent` | — | Existing AV latent to **continue from**, treated as a prior chunk. The accumulator is seeded with it and generation continues after it via the overlap mechanism — no masks, no re-sampling of the prior. `latents` then defines only the **new** region to generate. Output is prior + continuation. |
| `guiding_downscale_factor` | 1.0 | **Small-grid IC-LoRA reference factor** — wire from the LoRA Metadata Reader (or IC-LoRA Loader Model Only) so the LoRA's `reference_downscale_factor` metadata drives it (pixel spatial upscaler x2 = 2, x4 = 4). The guiding latent must then be `gen_dims/factor` — e.g. the stage-1 latent **directly** for a 2× pass. Each chunk's guide slice is dilated onto the full grid with RoPE-adjusted patch spans (the trained reference geometry); dilation-hole tokens are filtered before attention, so the reference overhead is only the small-grid token count. 1 = normal dense reference (Ingredients, CrossView, depth/pose). Spatial tiling >1×1 not supported with factor >1. |
| `optional_denoise_mask` | — | **Spatial inpaint mask** (white = regenerate, black = keep pinned to the input latent's video). The pack's primary inpainting path — **no inpaint LoRA needed**: the mask states "synthesize here / reproduce the rest" structurally, and the fill coheres because the model sees the pinned latent while denoising. Requires real video in the input latent (v2v; empty pins black). Single mask = static; batches (e.g. SAM per-frame) resample onto the latent grid. Merged keep-wins (min) with `video_cond_strength` / overlap / keyframe masks. Keep `video_cond_strength 0`. See the field guide §5b. IC-LoRA route (Inpaint Color Fill + inpaint LoRA) is the fallback for large holes / hard edits. |

### Continuing an existing video

Wire an encoded AV latent into `optional_prior_av_latent` and the looping sampler treats it as an already-generated prior chunk: it seeds the accumulator with it and generates a continuation using the same trained overlap continuity it uses between its own chunks. The prior is preserved (it is the accumulator's seed, never re-sampled) except its last `temporal_overlap` frames, which blend into the first new chunk for a seamless handoff.

```
existing video → VAE encode (video) ─┐
existing audio → VAE encode (audio) ─┴─ LTXVConcatAVLatent → optional_prior_av_latent
empty AV latent (new duration; scene_lengths for the continuation) → latents
```

- `latents` sizes and scripts only the **new** region (its length is the continuation length; `scene_lengths` and per-chunk prompts apply to the new content).
- The output is the full `prior + continuation` as one AV latent. No masks are involved on input or output.
- Works with `audio_cond_strength` for the new region (free-gen or condition on a new track); the prior's own audio carries forward via the audio bridge.
- Not yet combined with mid-video keyframes or IC-LoRA guiding latents in the same pass (their indices assume a from-zero timeline).
| `guiding_start_step` | 0 | Sigma schedule step at which guiding latents begin influencing. |
| `guiding_end_step` | 1000 | Sigma schedule step at which guiding latents stop influencing. |

### Audio continuity

Audio continuity across chunk boundaries is maintained through two complementary mechanisms:

1. **Noise mask carry-over** — the last N audio frames from the accumulated output are carried into each chunk's init latent with a strong conditioning mask (`audio_overlap_cond_strength`).
2. **Seam stitching** — how the accumulated tail joins the new frames depends on `audio_overlap_cond_strength`:
   - **≥ 1.0 (default): keep-carry-verbatim.** The accumulator's real audio tail is preserved and only the genuinely-new frames are appended. The carry was frozen (mask 0), so it must **not** be re-voiced under the incoming chunk's conditioning — regenerating it there causes the seam to "speak" the new chunk's text, static-image burn-in, and continuation gibberish. Keeping it verbatim eliminates all three.
   - **< 1.0: regenerated bridge.** The carry was allowed to drift, so the model's own regenerated carry-over replaces the accumulated tail (hard-joining the real tail would create a discontinuity).

   Both paths yield identical length; the carry length `(overlap − 1) × 8 + 1` absorbs the first-frame asymmetry, so chunk audio is stitched without trimming or padding.

For voice identity across chunks/turns, the designed mechanism is **carry-swap** — replacing an extend chunk's audio carry with a reference-voice latent (on-timeline, mask-frozen, discarded from output by the keep-verbatim stitch): see [SPEC_NEG_REF_AUDIO.md](SPEC_NEG_REF_AUDIO.md) (confirmed design, not yet built). The `ref_audio` conditioning pathway (`LTX Add Audio Latent Guide`, core `LTXVReferenceAudio`) only functions with the TalkVid ID-LoRA loaded.

### Audio alignment notes

- `AUDIO_LATENTS_PER_SECOND = 25.0` (fixed for LTX2.3 AV)
- LTX first-frame asymmetry: first video latent = 1 pixel frame, all subsequent = 8 pixel frames. Pixel frame count: `px = (T_v - 1) * 8 + 1`.
- Output is trimmed to exactly match the requested frame count after sampling. Any overshoot from the last chunk's tile window is discarded.

---

## LoRA Training

Two-phase LoRA training pipeline for LTX-2.3 (22B AV model) via [musubi-tuner](https://github.com/AkaneTendo25/musubi-tuner). Phase 1 trains visual character identity on images; Phase 2 warm-starts from that LoRA and trains audio layers on voice/sound clips. The two phases produce a single merged LoRA covering both visual and audio identity.

Requires musubi-tuner installed and configured with an LTX-2.3 checkpoint and Gemma text encoder.

---

### LTXAV Character LoRA Training

**Phase 1.** Trains a LoRA on a character's visual identity using static images. Outputs the trained LoRA path for use as the warm-start input to Phase 2.

Dynamic image slots — connect 1–20 `IMAGE` inputs and set `image_count` to match. Each slot has a `caption_N` text field for the image description.

| Input | Default | Description |
|---|---|---|
| `model` | — | Loaded LTX model |
| `workspace_dir` | — | Root folder for all training artifacts |
| `run_name` | `CharacterLoraTrainingRun` | Label for logs |
| `output_name` | `CharacterLoraTraining` | Prefix for output LoRA files |
| `image_count` | 4 | Number of active image slots (1–20) |
| `training_steps` | 1000 | Total training steps |
| `num_repeats` | 4 | Dataset repeats per epoch |
| `lora_target_preset` | `full` | Layer target (`full`, `attn`, etc.) |
| `network_dim` | 32 | LoRA rank |
| `network_alpha` | 16 | LoRA alpha |
| `learning_rate` | 1e-4 | Training learning rate |
| `blocks_to_swap` | 20 | CPU-offload blocks to reduce VRAM usage |
| `cache_strategy` | `auto` | `auto` / `force` / `skip` latent cache |
| `musubi_root` | — | Path to musubi-tuner installation |
| `ltx2_checkpoint` | — | Path to LTX-2.3 `.safetensors` checkpoint |
| `gemma_root` | — | Path to Gemma text encoder root |

**Outputs:** `model`, `latest_state_path`, `lora_path`, `log_path`, `video_filename_prefix`, `output_name`, `completed_steps`, `total_target_steps`

Connect `lora_path` → `base_lora_path` on the audio training node.

---

### LTXAV Audio LoRA Training

**Phase 2.** Continues training from the Phase 1 LoRA, targeting only the LTX-2.3 audio attention layers (`audio_attn`, `video_to_audio_attn`, `audio_ff`). Trains on short audio clips with captions. On completion, merges the audio-layer weights back into the character LoRA to produce a single unified file.

Dynamic audio slots — connect 1–20 `AUDIO` inputs (e.g. from VHS Load Audio) and set `audio_count` to match. Each slot has a `caption_N` text field.

| Input | Default | Description |
|---|---|---|
| `model` | — | Loaded LTX model |
| `base_lora_path` | — | `lora_path` output from Phase 1 |
| `workspace_dir` | — | Same root folder as Phase 1 (audio artifacts go in `audio_*` subdirs) |
| `run_name` | `AudioLoraTrainingRun` | Label for logs |
| `output_name` | `AudioLoraTraining` | Prefix for audio LoRA output files |
| `audio_count` | 8 | Number of active audio slots (1–20) |
| `training_steps` | 400 | Total training steps |
| `num_repeats` | 2 | Dataset repeats per epoch |
| `network_dim` | 32 | LoRA rank — must match Phase 1 |
| `network_alpha` | 16 | LoRA alpha — must match Phase 1 |
| `learning_rate` | 1e-4 | Training learning rate |
| `blocks_to_swap` | 0 | CPU-offload blocks |
| `audio_bucket_interval` | 2.0 | Audio bucket step size in seconds |
| `audio_bucket_strategy` | `pad` | `pad` (loss-masks padding) or `truncate` |
| `audio_only_target_resolution` | 64 | Latent geometry resolution for audio-only mode |
| `audio_only_sequence_resolution` | 64 | Virtual sequence resolution for noise schedule |
| `ltx2_audio_only_model` | `true` | Use the physically audio-only transformer variant |
| `cache_strategy` | `auto` | `auto` / `force` / `skip` latent cache |
| `musubi_root` | — | Path to musubi-tuner installation |
| `ltx2_checkpoint` | — | Path to LTX-2.3 checkpoint |
| `gemma_root` | — | Path to Gemma text encoder root |

**Outputs:** `model`, `latest_state_path`, `merged_lora_path`, `log_path`, `output_name`, `completed_steps`, `total_target_steps`

`merged_lora_path` is the single unified `.comfy.safetensors` combining Phase 1 visual weights and Phase 2 audio weights. Use this file for inference — no need to stack two LoRAs.

---

### Character Dataset Prompt Generator

Companion node for building Phase 1 image datasets. Generates non-repeating camera-angle prompts: 8 horizontal angles × 3 vertical angles × 6 framings (135 valid views — rear angles exclude face close-ups), each combined with a neutral pose, white studio backdrop, and flat lighting. Used-view history persists per character in a JSON file across workflow runs and restarts; when all views are exhausted it cycles. Never cached — every queue produces the next prompt.

| Input | Description |
|---|---|
| `character_name` | Key for the history file (one JSON per character) |
| `character_description` | Appearance text inserted into every prompt |
| `mode` | `sequential` picks views in order; `random` uses seed + run count |
| `seed` | Random mode only |
| `reset` | Clear this character's history and start over |

**Outputs:** `prompt`, `remaining`, `total`, `view_key`

---

## License

MIT — see [LICENSE](LICENSE).

Portions derived from or inspired by [comfyui-vrgamedevgirl](https://github.com/vrgamegirl19/comfyui-vrgamedevgirl) by vrgamegirl19 (MIT).
