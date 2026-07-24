# Changelog

## 1.1.0 — 2026-07-15

### Added
- **LTX Video Outpaint Latent** (2026-07-24, validated): latent-space outpaint
  prep for the base-model path — zero-pads an encoded video latent (real
  centre, **zero margin**) + emits a feathered denoise mask. The zero margin
  regenerates cleanly because it is empty; padding *pixels* (black/grey/green)
  bakes structured content the model preserves — that's why pixel-padded
  outpaint left borders. No LoRA, no fill color. `overlap` (full-regen band
  straddling the seam) + `feather` (ramp into the kept interior) are
  independent seam knobs; `margin_fill` zeros (default) or noise. Run in a
  full-denoise pass. Fixes the black-artifact failure, not the
  one-sided-context limit (moving-camera / simple margins strongest).
- **Video Cut Marker: start marker** (2026-07-22, confirmed): blue `S` marker
  trims the head; new `skip_first_frames` output (appended) wires to the VHS
  loader so generation begins there. The schedule is measured across the
  `[start, end]` window; the offset round-trips via a new appended
  `start_frame` widget. Both additions are append-only (existing links/values
  preserved).
- **Video Cut Marker: auto-cut placer + new-media reset** (2026-07-21,
  confirmed): an interval box + ⚡ button replaces all cuts with one every N
  seconds (grid-snapped, start→end); loading new media (combo change or
  upload) resets the schedule while the workflow-restore path stays untouched
  (F5 still restores). `Del`/`Backspace` no longer delete markers (they remain
  ComfyUI's delete-node); `X` / right-click / ✕ do.
- **Dimension Calculator 3 Stage: custom override** (2026-07-21, confirmed):
  `use_custom` + `custom_role` (`quarter (stage 1)` ×4 / `half (stage 2)` ×2 /
  `full (final)`) mirroring the base calculator, each role snapping to its own
  grid (÷32 / ÷64 / ÷128) so every derived stage stays LTX-valid. Appended
  optional inputs — existing graphs unchanged.
- **AV Looping Sampler: spatial denoise mask** (`optional_denoise_mask`,
  2026-07-22, validated): base-model inpainting with **no inpaint LoRA** —
  white = regenerate, black = keep pinned to the input latent's video. The
  mask states "synthesize here / reproduce the rest" structurally, so the
  IC-LoRA's role on the *where* is redundant; the fill coheres because the
  model sees the pinned latent while denoising (one scene, not a composite).
  Merged keep-wins (elementwise min) with `video_cond_strength` / overlap /
  keyframe masks; single mask static, batches resample onto the latent grid
  (SAM per-frame works directly); spatial tiling supported. Requires real
  video in the input latent. See field guide §5b. This is now the pack's
  primary inpaint path; the IC-LoRA route is the large-hole / hard-edit
  fallback.
- **LTX Inpaint Color Fill** (2026-07-22): solid-color mask fill for the
  IC-LoRA inpaint route — magenta / chroma green / Lightricks green presets +
  custom hex (core's `LTXVInpaintPreprocess` hardcodes one green), `binarize`
  for exact fills. Composite at final resolution to keep the boundary color
  exact.
- **LTX Streaming Video Encode** (2026-07-21/22, validated — long-latent
  roundtrip with no stitches): chunked VAE encode straight from a video file,
  the input mirror of the streaming save. Causal left-context per chunk
  (trimmed, incl. the malformed 1-frame head latent) makes it exact; only
  latents accumulate, RAM constant at any source length. Bookkeeping proven
  0-diff against single-pass reference across chunk sizes / mid-latent tails /
  short sources.
- **Streaming Save diagnosability** (2026-07-21): ffmpeg stderr is now
  captured (deadlock-safe via temp file) and included in every error, the
  resolved ffmpeg path is logged, a mid-stream ffmpeg death reports its own
  stderr instead of a bare BrokenPipeError, and `wav_tmp` is cleaned up on mux
  failure.
- **LTX AV Streaming Decode & Save** (2026-07-17/18, validated): chunked
  causal-exact VAE decode piped directly into ffmpeg — constant RAM at any
  video length; audio muxed from decoded AUDIO; inline player on finish.
  Slower than the normal decode path — use only where length requires it.
- **LTX LoRA Metadata Reader** (2026-07-18, validated): safetensors-header
  reader (no weight load). One combo drives loader (`lora_path` →
  `opt_lora_path`) and sampler (`latent_downscale_factor` →
  `guiding_downscale_factor`).
- **AV Looping Sampler: small-grid IC-LoRA references** (2026-07-17/18,
  validated): appended `guiding_downscale_factor` (FLOAT, metadata-wireable).
  Per-chunk guide dilation + RoPE patch-span adjustment — the trained
  reference geometry of the pixel spatial upscaler IC-LoRAs (x2/x4), enabling
  chunked long-form pixel upscaling. Factor 1 = unchanged dense references.

### Fixed
- **Conditioning sanitizer — stale guide bookkeeping stripped on entry**
  (2026-07-22): the sampler now removes any `keyframe_idxs` /
  `guide_attention_entries` from incoming conditioning (it builds its own
  guides per chunk) and no longer memoizes `raw_conds` onto the cached guider.
  Fixes the intermittent `guide pre_filter_counts != keyframe grid mask
  length` that "went away after a cache clear" — guide bookkeeping was
  accumulating on a ComfyUI-cached guider across queue runs. Prints
  `stripping stale guide conditioning …` when it acts.
- **Guide attention-entry registration is now measured, not predicted**
  (2026-07-17): pre_filter_count read from the actual keyframe_idxs delta, so
  registration survives core frame-accounting changes (fixes
  `guide pre_filter_counts != keyframe grid mask length` after the 2026-07
  ComfyUI update).
- **Video Cut Marker: state persistence across page refresh** (2026-07-18):
  media reloads with restored widget values via onConfigure, and restore no
  longer rewrites the saved schedule when the loaded media doesn't match
  (previously a refresh could silently destroy the schedule).
- **LTX Video Cut Marker (Scenes)** (2026-07-16/17, validated): interactive
  timeline widget — video/audio loading with waveform display, latent-grid-
  snapped scene cuts, optional end marker, time-anchored emit-fps math (24→25
  force_rate safe). Outputs `scene_lengths`, `frame_count`, `video_path`,
  `frame_load_cap`. Includes widget lifecycle management (RAF teardown on node
  removal, dirty-flagged rendering).
- **LTX Keyframe Planner** (2026-07-17, validated): `scene_lengths` →
  end-anchored keyframe indices (`0,120,248,-1` style) for keyframe-travel
  generation; each scene converges on its own destination image.
- **LTX Keyframe Pair Concat** (2026-07-17, validated): consecutive keyframe
  pairs as one composite (horizontal/vertical, divider gap) for vision-LLM
  scene-transition prompting; `index`-driven cycling with `total_pairs` bound.
- **Tiled Latent Upsampler: temporal mode** (2026-07-16, validated). Auto-detects
  the upscaler type from the first tile's output (`L → L` spatial, unchanged
  path; `L → 2L−1` temporal). Temporal tiles anchor at `2×` input position with
  malformed tile-head latents dropped (`head_trim`, new appended input) and
  crossfades in output coordinates. Previously the temporal upsampler crashed
  the tiled node with a tensor-size mismatch. See `SPEC_TILED_TEMPORAL.md`.
- **LTX AV Reference Audio Bank (Carry-Swap)** (`LTXAVReferenceAudioBank`): up to
  four reference voices + a per-chunk voice schedule (`1|2|1|2`) for turn-based
  dialog voice identity. No ID-LoRA required. See `SPEC_NEG_REF_AUDIO.md`.
- **AV Looping Sampler:** new appended optional input `optional_ref_audio_bank`.
  Per the bank's schedule, an extend chunk's frozen audio carry is replaced with
  the scheduled reference voice (sampling context only — the keep-verbatim
  stitch keeps the accumulator's real tail in the output). `swap_mode`
  `on_change` (turn seams) or `always` (testing/anti-drift).
- **LTX AV Cross-Attention Toggle** (`LTXAVCrossAttnToggle`): gate the AV
  model's a2v / v2a cross-modal couplings. `v2a_cross_attn = False` enables
  changing the spoken words over a guide video (the guide's visible lips
  otherwise out-muscle the text during audio generation → gibberish).

### Changed
- **Keep-carry-verbatim audio stitch:** at `audio_overlap_cond_strength >= 1.0`
  the accumulator's real audio tail is preserved and only genuinely-new frames
  are appended (the regenerated carry is discarded). Fixes seam re-voicing
  ("speaking the next chunk's prompt"), static-image burn-in, and
  continuation-boundary gibberish. Below 1.0 the previous regenerated-bridge
  behavior is retained. Carry-swap chunks always take the verbatim path.

### Documentation
- `LTXVAddAudioLatentGuide` marked **ARTIFACT** (retested: no effect without the
  ID-LoRA — the `ref_audio` negative-coordinate placement is an ID-LoRA training
  convention the base model is deaf to).
- `LTXAVReferenceAudioMulti` / `LTXAVSpeakerPromptProvider` marked **ABANDONED**
  (ID-LoRA is inherently single-voice).
- `SPEC_NEG_REF_AUDIO.md` added (carry-swap design, rejected routes, test plan).

## 1.0.0 — 2026-07-15
- Initial versioned state (pyproject.toml added for ComfyUI Manager/registry).
