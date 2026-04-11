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
Snaps a desired duration to the nearest valid LTX frame count — `(frames - 1) % 8 == 0` — and returns the actual snapped duration.

| Input | Description |
|---|---|
| `seconds` | Desired clip length |
| `fps` | Frames per second |

**Outputs:** `frame_count` (pixel frames), `latent_frames`, `actual_seconds`

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

### LTXAV Extend Mask
Combined audio+video mask node for LTX2 sliding-window generation. Extends latents with zero padding beyond `end_time` and applies linear ramp transitions at mask boundaries.

| Input | Description |
|---|---|
| `video_latent` | 5D video latent `[B,C,T,H,W]` |
| `audio_latent` | 4D audio latent `[B,C,T,F]` |
| `audio_vae` | Audio VAE (used to derive latent rate) |
| `video_fps` | Frames per second |
| `start_time` | Start of generation region (seconds) |
| `end_time` | End of generation region (seconds) |
| `pad_to_time` | Extend with zeros to this duration (0 = no pad) |
| `slope_len` | Ramp length in latent frames at each boundary |
| `strip_input_mask` | Strip existing mask before applying new one |

**Outputs:** `video_latent`, `audio_latent`, `output_video_seconds`, `audio_latents_per_second`
