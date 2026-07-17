# SPEC: Temporal-Mode Support for the Tiled Latent Upsampler

Status: **BUILT and user-validated** (2026-07-16 — "no fault found" in tiled
temporal upsampling; spatial regression unchanged by construction via mode
auto-detect)

## Problem
`LTXVAVLatentUpsamplerTiled` (`nodes/utils.py`) assumes the upscale model
preserves the temporal dimension: the accumulator and weight tensors are sized
at input `T`, and each upsampled tile is written back to *input* coordinates.
The LTX **temporal** upsampler returns `2L−1` latents for an `L`-latent tile
(observed: 16 → 31), so the write crashes
(`result[:, :, t_start:t_end] += up_tile` → 31 into a 16-slot window).

The non-tiled `LTXVAVLatentUpsampler` already handles temporal upscaling (no
accumulator — output shape is free). The tiled variant exists only for VRAM;
this spec makes it temporal-capable for long videos that OOM the full tensor.

## Latent math (grounded)

- Latent count: `T` latents ⇔ `(T−1)·8+1` pixel frames. Temporal 2× doubles
  the pixel timeline: `px' = 2·px`, giving `2((T−1)·8+1)−1` pixel frames
  ⇒ **`T_out = 2T−1` latents**. Matches the observed 16 → 31.
- **Anchor mapping:** input latent `t` (global) ⇔ output latent `2t` (global).
  A tile of `L` latents starting at global `t0` produces `2L−1` output latents
  landing at **global output `[2·t0, 2·t0 + 2L−1)`**.
  - first tile `t0=0, L=16` → out `[0, 31)` ✓
  - next tile `t0=12` (overlap 4) → out `[24, 55)`; overlap with previous tile
    in output space = `[24, 31)` = **`2·ov − 1` output latents**.
- Global output buffer: `2T−1` latents; weights likewise.

## The first-frame asymmetry landmine (why naive blending is wrong)
A tile sliced from mid-video and fed to the upsampler standalone is treated as
a *video start*: its first output latent is the 1-pixel-frame start latent, and
the tile head generally lacks left temporal context. In the global assembly,
position `2·t0` of a non-first tile must be a regular 8-px latent — the tile's
head content is malformed for its global position (same class of problem as the
sampler's stitch, which drops local frame 0 via `video_trimmed[1:]`).

**Mitigation — head trim:** for every non-first tile, drop the first
`head_trim` output latents (default **2**, minimum **1** — latent 0 is always
malformed) and let the previous tile own that region. Crossfade over the
remaining output overlap:

```
usable blend span = (2·ov − 1) − head_trim      # default: 8−1−2 = 5 latents
```

Validation: require `2·ov − 1 − head_trim ≥ 1` (with defaults ov=4, trim=2:
fine). Non-last tile *tails* also lack right context, but the incoming tile's
(post-trim) blend covers that region — no separate handling.

## Design

Modify `LTXVAVLatentUpsamplerTiled` in place — **auto-detect the mode from the
first tile's output shape**, so existing spatial workflows are bit-identical
and the temporal case goes from crash to working (strictly additive):

- `T_out_tile == L`        → spatial mode: existing code path, untouched.
- `T_out_tile == 2L−1`     → temporal mode: new math below.
- anything else            → clear error naming both supported mappings.

Temporal mode:
1. Allocate `result`/`weights` at `2T−1` (H/W from the tile as today — also
   covers a hypothetical combined spatial+temporal model for free).
2. Per tile at `t0` with output `up_tile` (`2L−1` latents):
   - `g0 = 2·t0`; if `t0 > 0`: drop `head_trim` head latents,
     `g0 += head_trim`.
   - Weight vector over the written span: linear ramp-in over the usable blend
     span when `t0 > 0`; ramp-out over `2·ov − 1` at the tail when not final
     (mirrors existing weighting, in output coordinates).
   - Accumulate into `result[:, :, g0 : g0 + written]`.
3. Normalize by weights as today.
4. Audio passthrough unchanged — temporal upscale doubles fps at the same
   duration, so the audio component stays time-matched by construction.

New input, **appended** (widget-safe): `head_trim` (INT, default 2, min 1,
max 8) — "Output latents dropped from each non-first tile's head in temporal
mode (tile heads are malformed video-start latents). Ignored in spatial mode."

Versioning: MINOR (new appended input + previously-crashing path now works;
spatial behavior unchanged).

## Notes & caveats
- The `t ⇔ 2t` anchor is the pixel-math ideal and matches the observed shape;
  the model's learned interpolation should follow it, but **seam quality is
  empirical** — `head_trim` is the tuning knob if tile joins show motion
  stutter (raise it; each +1 costs 1 latent of blend span, so raise `ov`
  along with it).
- Downstream: temporal-upscaled output is 50 fps material — single-shot
  refinement only; the looping sampler's audio math is 25-fps-only
  (see `SPEC_50FPS.md`).
- The tiled path remains statistically inferior to the full-tensor node
  (GroupNorm stats per tile); same guidance as spatial — follow with a
  low-sigma refinement pass. Use the non-tiled node whenever VRAM allows.

## Test plan
1. Spatial regression: existing workflow, before/after — outputs must be
   bit-identical (mode detection must not perturb the spatial path).
2. Temporal, tiled vs non-tiled on a short clip that fits both: compare
   seams at tile joins (the non-tiled output is ground truth). Sweep
   `head_trim` 1/2/4.
3. Long-video OOM case (the motivating one): confirm it completes and the
   joins survive the low-sigma refine.
