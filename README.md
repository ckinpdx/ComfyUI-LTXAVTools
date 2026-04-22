# ComfyUI-LTXAVTools

Utility nodes for LTX2 audio-video generation workflows in ComfyUI.

---

## Nodes

### LTX Dimension Calculator
Aspect-ratio-aware resolution picker. Outputs only LTX-compatible resolutions (divisible by 64). Dynamic dropdown updates when ratio or orientation changes.

**Outputs:** `width`, `height`, `width_half`, `height_half`, `label`

### LTX Dimension Calculator 3 Stage
Same as above but constrained to div-by-128 resolutions, compatible with 3-stage pipelines (full / half / quarter resolutions all remain div-by-32).

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

---

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

---

### LTX Add Audio Latent Guide
Injects a raw audio latent as reference conditioning for LTX2 AV generation. The audio is placed at negative temporal RoPE positions (before t=0) so it influences audio character without contaminating the generated latent sequence.

Input must be a raw 4D audio latent `[B, C, T, F]`. Use `LTXVSeparateAVLatent` first if you have a combined AV latent.

| Input | Description |
|---|---|
| `positive` | Positive conditioning |
| `negative` | Negative conditioning |
| `audio_guide_latent` | 4D audio latent to inject as reference |

**Outputs:** `positive`, `negative`

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

## LTX AV Looping Sampler

Temporal (and optionally spatial) tiling sampler for long-form video+audio generation with the LTX2 AV model. Generates video and audio jointly as a NestedTensor latent across multiple overlapping chunks, accumulating a coherent sequence longer than any single context window.

Input latent must be an AV NestedTensor (e.g. from `LTXAudioOnlyLatent` or an AV encode). Use `LTXVLoopingSampler` for video-only generation.

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
| `temporal_tile_size` | 80 | Pixel frames per temporal chunk. Must satisfy `(size - 1) % 8 == 0`. |
| `temporal_overlap` | 24 | Pixel frames of overlap between chunks. The overlapping region from the previous chunk is injected as a guide so the model maintains visual continuity. |
| `guiding_strength` | 1.0 | Conditioning strength for guiding latents (IC-LoRA / latent guides). |
| `temporal_overlap_cond_strength` | 0.5 | Noise mask strength for the video carry-over (overlap) region. Higher values hold the overlap more rigidly; lower values let the model blend more freely. |
| `audio_overlap_cond_strength` | 0.9 | Noise mask strength for the audio carry-over region. Try 0.9–1.0 if chunk boundaries sound rough. |
| `cond_image_strength` | 1.0 | Noise mask strength for image keyframe conditioning. |
| `horizontal_tiles` | 1 | Number of spatial tiles horizontally. |
| `vertical_tiles` | 1 | Number of spatial tiles vertically. Audio is accumulated from tile (0,0) only. |
| `spatial_overlap` | 1 | Latent-space pixels of spatial overlap between tiles. |
| `video_fps` | 25.0 | Must match the fps of the AV latent. Used for audio frame alignment. LTX2 AV is trained at 25 fps. |

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
| `guiding_start_step` | 0 | Sigma schedule step at which guiding latents begin influencing. |
| `guiding_end_step` | 1000 | Sigma schedule step at which guiding latents stop influencing. |

### Audio continuity

Audio continuity across chunk boundaries is maintained through three complementary mechanisms:

1. **Noise mask carry-over** — the last N audio frames from the accumulated output are carried into each chunk's init latent with a strong conditioning mask (`audio_overlap_cond_strength`).
2. **ref_audio guide tokens** — the carry-over frames are reshaped into token format and injected as `ref_audio` conditioning, so the model attends to the spectral identity of the previous chunk.
3. **Bridge stitching** — the stitched audio uses the model's own regenerated carry-over as the join point rather than a hard latent-space concatenation, avoiding spectral discontinuities at boundaries.

### Audio alignment notes

- `AUDIO_LATENTS_PER_SECOND = 25.0` (fixed for LTX2 AV)
- LTX first-frame asymmetry: first video latent = 1 pixel frame, all subsequent = 8 pixel frames. Pixel frame count: `px = (T_v - 1) * 8 + 1`.
- Output is trimmed to exactly match the requested frame count after sampling. Any overshoot from the last chunk's tile window is discarded.
