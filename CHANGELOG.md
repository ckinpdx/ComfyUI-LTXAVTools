# Changelog

## 1.1.0 — 2026-07-15

### Added
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
