# Changelog

## 1.7.0 - 2026-07-30

### Added
- **LTX AV Add IC References (Multi)** — per-speaker IC-LoRA references, the
  visual counterpart to `LTXAVReferenceAudioMulti`. `reference_images_n` are
  injected only on chunks tagged `[SPEAKER n]`; `all_chunks_images` (a
  background plate) go into every chunk and are merged AFTER the selected
  speaker's views, keeping the tail slot a hand-built batch conventionally
  gives them — MSR is positional, so that is not cosmetic.

  Two wins over sending every character's views to every chunk: the prompt
  stops being the only thing arbitrating who is on screen (a wardrobe phrase is
  a weak handle against a face the model can see), and reference tokens — which
  are paid PER CHUNK — drop with the number of views. Two characters at 4 views
  plus a background is 9 views a chunk unscheduled, 5 scheduled.

  An untagged chunk gets EVERY speaker's views, not just the shared plate: no
  tag means we do not know who is on screen, and an establishing shot may hold
  everyone. A tagged chunk with no matching bank entry warns and falls back.

### Changed
- The encode path is now shared. `LTXAVAddICReferences` and the Multi node call
  one `encode_reference_pack()`, so the grid math, the 8n+1 truncation assert
  and the token accounting exist in exactly one place. The single node's
  behaviour and widgets are unchanged.
- Packs merge along the time axis into ONE guide append rather than several —
  cheaper, and it keeps a single deterministic slot order. Packs built for
  different geometry are refused with a message naming both, instead of
  producing an opaque shape error later.

## 1.6.1 - 2026-07-30

### Fixed
- **The looping sampler destroyed the base conditioning after chunk 0.**
  `_prepare_guider` did `new_g = copy.copy(guider)` — a SHALLOW copy, so
  `new_g.original_conds` was the *same dict object* as the incoming guider's, and
  `CFGGuider.set_conds` assigns into it. Chunk 0's `set_conds` therefore
  overwrote the real base conditioning, and every later chunk read chunk 0's own
  per-chunk conds back as "base". `new_g` now gets its own `original_conds` dict.

  Blast radius: ANY conditioning-level key set upstream that the sampler does not
  explicitly re-carry per chunk was silently lost from chunk 1 onward. The keys
  that are carried (`ref_audio`, `frame_rate`, `ic_reference_pack`) appeared to
  work only because chunk 0 handed them back on the next read.

  The symptom that exposed it: per-chunk `[SPEAKER n]` voice switching never
  fired — `LTXAVReferenceAudioMulti`'s `ref_audio_bank` is not carried per chunk,
  so it vanished with chunk 0's `set_conds` and every tagged chunk silently fell
  back to the default voice.

### Added
- When a `[SPEAKER n]` chunk falls back to the default voice, the log now says
  WHY — bank key absent (with a dump of the keys actually present on the guider's
  conditioning) vs bank present but empty. The first distinguishes a wiring
  problem from a node problem; the key dump is what located the bug above.

## 1.6.0 — 2026-07-30

### Fixed
- **Chunk 0 no longer clamps the audio length — it pads, like the extend chunks
  already did.** `T_a` now stays `audio_pos(T_v, fps)` unconditionally; a short
  input audio track is zero-padded rather than shortening the chunk. Clamping
  broke the invariant the whole boundary-map scheme rests on (`T_a ==
  audio_pos(T_v, fps)`) and made chunk 0 the only builder that disagreed with
  the rest. Verified delta 0 across empty / exact / short / long / 50 fps.
- **Warns when the input audio is shorter than the timeline**, naming the
  shortfall in seconds and whether `audio_cond_strength` will freeze the padded
  zeros. A zero audio latent is out-of-distribution — it decodes to a hum, not
  silence — so at high strength the hum is baked into the output with nothing
  else to indicate it.

### Changed
- **`audio_cond_strength` is now a mask control only.** The audio init is seeded
  from the input latent whenever input audio exists, independent of the
  strength — matching video, where `video_init` always holds the input and the
  mask decides what regenerates. Previously one scalar set both, which made
  "input audio as init WITH a free span" inexpressible, i.e. exactly what a
  video denoise mask does by default. `LTXVEmptyLatentAudio` returns zeros, so
  free-generation paths (including the ID-LoRA's) take the same branch as before.

### Added
- **`optional_audio_denoise_mask`** — overrides the audio profile that is
  otherwise derived from `optional_denoise_mask`, for the case where the two
  modalities genuinely should differ (re-dubbing a span while the video stays
  untouched). Unconnected, audio follows video as before.
- **LTX AV Time Range Mask** — denoise mask from time ranges ("2-4, 7-9.5"),
  reporting the latent and audio spans the range actually snaps to.
- **Cut Marker: `start_seconds` / `duration_seconds` outputs** (appended) for
  audio loaders — the scheduled window in seconds, so `VHS_LoadAudio`'s
  `seek_seconds` / `duration` cover exactly the scheduled region.

## 1.5.0 — 2026-07-28

### Added
- **Temporal AUDIO masking in the AV Looping Sampler.** The audio mask was
  authored from two scalars (`audio_cond_strength` / `audio_overlap_cond_strength`)
  and so could only be uniform or split at the overlap — there was no way to say
  "freeze this span of audio, regenerate that one". It is now modulated by a
  per-frame profile **derived from the video denoise mask**, never authored
  separately, so the two modalities cannot disagree about where a frozen span is.
  - Mapping is the boundary map: video latent `t` owns audio
    `[audio_pos(t), audio_pos(t+1))`, with `audio_pos(0)` taken as 0 — latent 0
    owns one audio frame, every later latent owns `q = 200/fps`. Spatial
    reduction is MAX: if any part of a video frame regenerates, its audio does
    too. Verified exact at 25 and 50 fps.
  - Composed by MULTIPLICATION, so the profile can only ever freeze. The
    strength scalars still govern how hard input audio is held wherever the
    profile leaves it free, and the overlap carry keeps its own behaviour.
  - This is what makes temporal AV inpainting expressible — freeze A and B,
    regenerate the span between them, audio included.
- **Input validation on every clip node** (Join / Compose / Trim).
  - **fps is read from the clips and the LOWEST wins (`fps = 0`, the default).**
    Audio runs at 25 latents/second regardless of video rate, so
    `audio_pos(T_v, fps) == T_a` pins each clip's rate — `infer_av_fps` asks the
    clip instead of trusting a widget. Joining 24 + 25 resolves to 24 with no
    comparison nodes in the graph.
    - **Video is never touched** — a clip is T latents whatever rate you call
      it. Only the audio is retimed, by linear interpolation of the latent
      sequence, which is the latent-domain equivalent of a time stretch (~4%
      for 24↔25). Reported on every use; it is an approximation, not a
      resampler.
    - Candidate ORDER is preference, because short clips fit more than one rate.
      Without it a 24 fps clip resolves to **23.976** and every downstream
      duration inherits the drift — found in testing.
  - **fps was ALSO the source of an off-grid bug.** Bridge Compose allocated
    section audio as `k * q`, which only holds when `q = 200/fps` is an integer
    — so it broke at 24 fps. Now every section is a DIFFERENCE of `audio_pos`,
    which telescopes to the exact total at any rate (SPEC_50FPS's principle,
    applied where I had missed it).
  - **fps was verified against the clip itself, not trusted.** Audio runs at 25
    latents/second regardless of video rate, so `audio_pos(T_v, fps) == T_a`
    pins the rate — each clip is *asked* what it is (`infer_av_fps`). Feeding
    50 fps clips with fps set to 25 now fails immediately, naming the detected
    rate, instead of silently scaling everything by 2. Unique for any clip of
    realistic length; ambiguous only below ~5 latents between near-identical
    rates (23.976/24, 48/50). Desynced streams that match no standard rate are
    reported as such.
  - **Resolution is reconciled, aspect is not.** Joining clips at different
    RESOLUTIONS is the normal case — the same shot at two sizes, a stage-1 and a
    stage-2 output — so matching aspects downscale to the smallest automatically
    (`on_size_mismatch`, default `downscale_to_smallest`, set `error` to refuse).
    Differing ASPECT always errors: there is no correct answer without cropping
    or letterboxing, and both change the framing, so it stays the operator's
    call. Tolerance 2% absorbs latent-grid rounding.
    - The target is the smallest clip's OWN grid, not (min height, min width)
      independently, which could otherwise synthesise a grid matching no clip's
      aspect.
    - Latent-grid resampling is lossy and says so on every use — decode, resize
      in pixel space and re-encode when the detail matters.
  - **Manual fps still validates.** Setting a rate explicitly checks every clip
    against it and errors naming the detected rate, so a wrong widget can't
    silently scale the result.
- **LTX AV Bridge Compose / LTX AV Bridge Extract** — a generated transition
  with BOTH boundary conditions as real AV.
  - Compose emits `[A_tail | free bridge | B_head]` plus its denoise mask.
    Audio allocation keeps **both seams exact** — A_tail end-aligned,
    B_head start-aligned — pushing the unavoidable first-frame discrepancy to
    the composite's outer edges, which are frozen context.
  - Extract discards the scaffolding tails and rejoins the bridge between the
    **original full-resolution** clips, deliberately keeping one extra head
    latent so Join has a stub to consume and no bridge content is lost.
  - The `plan` handoff (`LTXAV_BRIDGE_PLAN`) carries the section counts, so all
    latent cutting stays internal to the nodes.
  - Supersedes **Bridge Prep**'s approach of pinning B with a single still: the
    bridge now meets B's actual motion and audio at generation time rather than
    at the refinement pass.

## 1.4.0 — 2026-07-28

### Added
- **LTX AV Trim Latents** — keep or drop a span from either end of an AV latent,
  slicing audio to match via `audio_pos`. Modes `keep_head` / `keep_tail` /
  `drop_head` / `drop_tail`, in seconds or exact latents. Head slices are exact
  (they keep latent 0, the genuine video-start latent); tail slices are aligned
  at their END, where a join or continuation happens.
  - Needed for the second-pass composite (`A_tail | bridge | B_head`) and for
    isolating a bridge from a sampler's prior+bridge output. 1.3.0 shipped
    `a_tail_seconds` without it, so that path dead-ended.
- **Bridge Prep: `trim_prior_latents` output** (appended). Slices have no stub
  latent, but Join drops one from every clip after the first — so trimming the
  prior off *exactly* would eat a latent of bridge. This value is
  `prior_latents - 1`, deliberately leaving the prior's final latent in place so
  Join has a stub to drop and the bridge survives whole. Verified end to end:
  identical final length at every `a_tail_seconds`, at 25 and 50 fps, with clip
  A and clip B content intact at the ends.

## 1.3.1 — 2026-07-27

### Fixed
- **Cut Marker: `emit_fps` driven by a link is now resolved in the timeline
  panel.** A linked widget's `.value` goes stale — the frontend never sees what
  the backend will resolve — so the panel previewed at the widget's old rate
  while the run used the link's. Since `scene_lengths` are pixel frames, that
  scaled every scene silently. The panel now walks one hop upstream (through
  Reroutes) and reads the value when the source is a literal.
  - **Only literal sources are trusted** (`Primitive` / `*Constant` / `Int` /
    `Float`). Reading "the first positive number" off an arbitrary node would
    return a step count or a seed and report it as the frame rate; a confident
    wrong answer is worse than an honest one. A computed source is reported as
    unresolved in the readout, with the rate the panel is previewing at.
  - The rate is now re-resolved on draw / readout / schedule-sync rather than
    only in the widget callback, so a linked rate is picked up without a
    callback ever firing — and a change re-anchors the start/end markers via
    the same path added in 1.2.1.
  - **Also re-resolved at queue time**, via `serializeValue` on the
    `scene_lengths` widget. Nothing in the panel runs while idle — draw and
    readout fire only on pointer/key/playback events — so changing a linked
    rate and hitting Run *without touching the timeline* would otherwise
    serialize a schedule built at the old rate. `scene_lengths` is widget 1 and
    `start_frame` is widget 3, so the re-sync's `start_frame` update lands
    later in the same serialization pass.

## 1.3.0 — 2026-07-27

### Added
- **LTX AV Join Latents** — concatenates 2-4 AV latents with the first-frame
  correction. A plain `cat` is wrong by a CONSTANT `+7` pixel frames and `+1`
  audio latent per join: clip B's latent 0 was encoded as a video start holding
  one pixel frame and decodes as eight mid-sequence. Dropping each subsequent
  clip's stub video latent and first audio latent gives `joined_T = ST - (N-1)`,
  and the total is reconciled against `audio_pos(T, fps)` rather than trusted.
  Verified exact for 2/3/4 clips at 25 and 50 fps.
  - Emits a temporal `region_mask` (`free_clips` + `mask_feather`) for refining
    one clip in place — the second-pass lever for bridges.
- **LTX AV Bridge Prep** — stages a generated transition between two clips.
  Head pinned by real AV via `optional_prior_av_latent`, tail by clip B's first
  frame at index `-1`. LTX already does the bridging; this only assembles the
  tensors. `a_tail_seconds` defaults to 2.0 (`0` = all of A).
  - Tail slices are aligned at their END: a mid-sequence latent owns 8 audio
    frames but a standalone clip's latent 0 owns 1, so any slice carries a
    surplus. Taking the last `audio_pos(k)` frames puts that discrepancy at the
    prior's head (discarded context) instead of at the seam.

## 1.2.1 — 2026-07-27

### Fixed
- **Cut Marker: start/end markers now follow wall-clock when `emit_fps`
  changes.** Cuts store a time (`c.t`) and re-anchor for free, but `startLat`
  and `endLat` are **latent indices** and were left untouched by the fps
  callback — so switching 25 -> 50 doubled the timeline under them and both
  markers silently landed at *half* their former time. With an end marker set
  that clamped the whole schedule to the first half of the video, with no
  error. The callback now snapshots both markers' times at the OLD rate and
  converts them back at the new one, alongside the existing cut re-snap.
  Verified: 25 -> 50 -> 25 round-trips to the exact original latent indices,
  and wall-clock is held to within one latent (the grid resolution).
  - Unaffected: graphs with no start/end marker were already correct, since
    cuts were always time-anchored.
- **Stale fps docs corrected.** The `emit_fps` tooltip and the Scene Length
  Calculator's `fps` row both asserted 25 as *the* LTX AV rate; both now state
  that they must equal the sampler's `video_fps`, and that a mismatch halves or
  doubles every scene silently. The tiled upsampler section claimed temporal
  (50 fps) output was "single-shot refinement only (the AV Looping Sampler's
  audio math is 25 fps)" — untrue since the `audio_pos` refactor, and now
  documents the round-trip including the `frame_rate` conditioning requirement.

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
