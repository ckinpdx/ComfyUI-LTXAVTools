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

### LTX Detail Sigmas
Generates a sigma schedule using a high-cluster + cliff + power-tail structure, designed for LTX distilled models. Concentrates steps near sigma=1.0 for fine detail establishment, then uses a single large cliff step (the distilled shortcut), followed by a power-curve tail to zero.

At 8 steps with defaults approximates the known community schedule: `1.0 → ~0.975 (linear cluster) → 0.65 (cliff) → ~0.0 (power tail)`.

| Input | Default | Description |
|---|---|---|
| `steps` | 8 | Total denoising steps |
| `high_fraction` | 0.5 | Fraction of steps in high-sigma cluster |
| `cluster_width` | 0.025 | Sigma range of cluster (1.0 down to 1.0 − width) |
| `cliff_sigma` | 0.65 | Landing sigma after the cliff step |
| `tail_power` | 2.0 | Power curve exponent for tail (higher = more steps near cliff) |

**Output:** `SIGMAS`
