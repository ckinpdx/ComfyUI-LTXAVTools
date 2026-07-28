# Changelog

## 1.2.0 — 2026-07-27

### Added
- **LTX LoRA Metadata Reader: `factor_int` output.** The reader emitted the
  downscale factor only as FLOAT, but several consumers are INT-typed —
  notably `LTXVDilateLatent`'s `horizontal_scale` / `vertical_scale` — and
  ComfyUI will not link FLOAT into an INT input, so the factor appeared to be
  "not picked up" downstream with no error. Now emitted as both.
  - **Appended, not inserted.** ComfyUI links outputs by INDEX; adding the INT
    beside the FLOAT would have silently re-pointed every existing `metadata`
    link in saved workflows. Existing graphs are unaffected.
  - Still FLOAT for `guiding_downscale_factor` (AV Looping Sampler) and
    `latent_downscale_factor` (LTX AV Add IC References) — both are FLOAT
    inputs and unchanged.

## 1.1.0 — 2026-07-15

### Added
- **LTX AV Add IC References** (2026-07-27, verified in use; §4.1 tensor diff
  exact, MSR end-to-end confirmed). Out-of-band IC-LoRA reference views for the
  looping sampler — identity conditioning with no timeline position that must
  be present in every chunk. The node encodes once and attaches an **inert**
  `ic_reference_pack`; the sampler injects it per chunk *after* conditioning
  assembly, which is what makes it survive MultiPromptProvider (§2.4). It does
  nothing on a stock sampler, by design.
  - **Convention verified by tensor diff, not by eye.** `apply_ic_references()`
    was diffed field-by-field against `LTXAddVideoICLoRAGuide` across
    {1,3,4} views × factor {1,2,4}: appended latent region, `noise_mask`,
    `keyframe_idxs` (start *and* end — the small-grid RoPE end-offset lives
    there), and the `guide_attention_entries` record. Exact match, maxdiff 0.
  - **Fixes a silent-truncation trap.** The temporal VAE keeps `((N−1)//8)*8+1`
    pixel frames, so a batch of 4 reference views encodes **1** and discards
    three with no error. `layout=one_latent_per_view` stacks them as
    `[v0, v1×8, …]` = `8(N−1)+1` so every view survives on its own latent frame;
    the node then asserts the encoded count rather than trusting it.
  - Implemented as a thin caller of the sampler's existing `_add_latent_guide`
    rather than a reimplementation — one copy of the convention, so core's
    frame accounting can change under it in one place.
  - **`layout=as_sequence` unlocks MSR.** The Licon MSR node emits an
    already-latent-aligned reference video (subjects on whole latent frames,
    background last: 41 frames -> 6 latents). It must be encoded as-is;
    `one_latent_per_view` would read its 41 frames as 41 views, destroying the
    multiplex and costing ~7x the tokens. Verified in use — this closes the
    2026-07-13 finding that long-form MSR was blocked on per-chunk injection.
  - v1 limits: single spatial tile (warns and skips under tiling), one pack.
- **Frame-rate mismatch check** — see Fixed, below.
- **50 fps audio support — `audio_pos` global boundary map** (2026-07-27,
  SPEC_50FPS implemented; validated `delta 0` at 25 fps regression and at
  50 fps with conditioned audio). Every audio count in the sampler is now a
  **difference of one global boundary map** rather than a chunk-local span:
  `a_carry = audio_pos(ov)`, `a_new = audio_pos(e) − audio_pos(s)`. Because
  `audio_pos(n) = (n−1)·q + floor(q/8 + 0.5)` for `q = 200/fps`, the rounding
  term is constant and cancels in any difference — so lengths telescope and
  cumulative drift is impossible. Previously every chunk-local span at 50 fps
  landed on exactly x.5 and rounded by parity (±1 audio frame per boundary,
  accumulating). Migrated: sampler chunk-0 / extend / debug / final trim,
  `LTXAVLatentCheck`, `LTXAVSeparateCheck`, `LTXAVExtendLatent`. At 25 fps all
  values are algebraically identical — outputs unchanged.
  - **Bonus:** off-grid rates (24, 30) no longer *accumulate* drift either —
    they retain only a bounded ~20 ms per-boundary quantization. 24 fps went
    from broken to workable.
  - **`video_fps` now warns** when `200/fps` is not an integer, listing the
    exact rates (1, 2, 4, 5, 8, 10, 20, 25, 40, 50, 100, 200).

### Fixed
- **`frame_rate` now carried onto per-chunk prompts** (2026-07-27): with
  MultiPromptProvider the per-chunk conditionings are bare text encodes and
  carry no `frame_rate`, which `model_base.py` defaults to **25 per
  conditioning** — so the positive branch ran at 25 while the negative (via
  `LTXVConditioning`) ran at the real rate, putting the two CFG branches on
  different time axes. `_prepare_guider` now carries it, mirroring the
  `ref_audio` fix. Invisible at 25 fps; desyncs at any other rate.
  - Related, not a code bug: **a graph with no `LTXVConditioning` node runs the
    model at 25 fps no matter what the sampler widget says** — the sampler's
    `video_fps` drives audio arithmetic only, while the model's temporal RoPE
    reads `frame_rate` from conditioning.
- **Frame-rate mismatch is now caught before the render, not after** (2026-07-27).
  The failure above is silent at render time: the audio is internally consistent
  (Latent Check reports `delta 0`, because it *is* correct for the sampler's
  rate) while the video is paced for 25 — you only find out on playback, an
  entire generation later. The sampler now inspects the guider's conditioning up
  front and warns on four shapes: conditioning carries no `frame_rate` while
  `video_fps ≠ 25` (missing `LTXVConditioning`); conditioning `frame_rate` ≠
  `video_fps` (reports the exact drift factor); positive and negative on
  different rates; positive set but negative defaulting. Diagnostic only —
  nothing is mutated, and an unrecognized guider shape is skipped silently.

### Added
- **Subject removal nodes** (2026-07-24, validated on basic scenes):
  - **LTX Removal Encode** — one-node prep: tight pixel noise-fill → VAE encode
    → latent zero, on one mask, with the recipe locked (grow/feather 0, decoded
    noise @ 0.1, fixed seed; only `temporal_mode` exposed). Output `(latent,
    mask)` → concat audio to `latents`, grow+blur the mask to
    `optional_denoise_mask`.
  - **LTX Noise Fill** — the pixel-space step alone: composites VAE-decoded
    (on-manifold) or gaussian noise into the masked region *before* encoding,
    so the subject is gone from the latent entirely and the VAE can't smear it
    from kept cells back into the hole. (The latent-zero-only path,
    **LTX Inpaint Latent**, leaks for removal because the encode already
    carried and smeared the subject — use it for in-place edits, not removal.)
  - **Mask `temporal_mode`** on LTX Inpaint Latent / LTX Removal Encode, and
    **`denoise_mask_temporal_mode`** on the AV Looping Sampler (match them):
    how per-frame masks reduce onto the 8:1 latent grid — `max` (union, safe,
    blobs at cuts), `last` (causal-aligned crisp, best across cuts),
    `mean_threshold`, `min`. Known limit: the VAE blends two contexts into one
    latent frame at a cut/fast motion, which no mask mode can un-blend
    (per-shot encode is the real cure, not yet built).
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
- **Mask → latent downsampling now uses max, not bilinear** (2026-07-24): the
  shared `ltx_mask_to_latent` reduced the temporal axis with max but the 32×
  *spatial* axis with bilinear — wrong for a coverage mask. Bilinear
  point-samples that reduction, so a solid region covered 77 latent cells where
  max covers 91 (the whole boundary ring read ~0 and was silently treated as
  KEEP → pinned to the init → a grey ring at the mask edge once the hole is
  noise-filled), and thin features came out ~50% fractional (any cell below 1.0
  has the init re-blended into it on *every* sampling step, per
  `samplers.py:637-641`). Now `adaptive_max_pool2d`: a latent cell is masked if
  the subject touches it at all. Fixes both the latent-zero nodes and the
  sampler's `optional_denoise_mask` in one place.
- **Docs correction — flow-matching init semantics**: earlier notes claimed the
  init is "~half the signal at σ≈1.0". That is the eps-model formula; LTX is
  flow-matching (`model_sampling.py:97`, `sigma*noise + (1-sigma)*latent_image`)
  so at σ = 1.0 the init contributes exactly zero. The init matters because it
  is re-blended every step wherever the denoise mask is below 1.0 — same
  conclusion (zero the hole), correct reason.
- **Streaming Video Encode: frame resampler now duplicates as well as drops**
  (2026-07-24): the old accumulator only decimated (force_rate < native), so
  for the upsample case (24 → 25) it passed native frames through 1:1 and
  `skip_first_frames` / `frame_load_cap` — emitted by the Cut Marker in
  emit_fps (25) space — were applied at the native rate, landing the encode
  ~`skip × (1/native − 1/out)` seconds late (e.g. skip 3401 started at 141.7s
  instead of 136.0s). Replaced with a drift-free nearest/hold index map
  (`out frame j ← native floor(j·native/out)`) that duplicates on upsample and
  drops on downsample, matching the VHS force_rate stream so the same indices
  hit the same content across the loader and the encode. Requires the encode's
  `force_rate` set to the emit/consume rate (25), same as VHS.
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
  the tiled node with a tensor-size mismatch. See `SPEC_TILED_TEMPORAL`.
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
