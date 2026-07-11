# LTX AV Looping Sampler — Field Guide

How to drive the sampler's inputs together to achieve different ends. The README documents
what each input *is*; this guide documents what each input *does to the generation* and
which combinations produce which behaviors. Everything here comes from mechanism-level
analysis of the LTX2.3 AV model and empirical long-form testing.

---

## 1. Mental model

The sampler generates a long video as a chain of overlapping chunks. Each chunk is a full
joint audio+video generation whose conditioning is assembled from up to six sources:

| Source | Input(s) | Reaches the model as |
|---|---|---|
| Text | guider conditioning / `optional_positive_conditionings` | cross-attention (video and audio branches) |
| Previous chunk | `temporal_overlap` region of the accumulated output | in-timeline latents + noise mask (video **and** audio) |
| Images | `optional_cond_images` + indices | I2V pin (index 0) or keyframe guides |
| Input audio | audio component of `latents` + `audio_cond_strength` | in-timeline frozen audio latents |
| References | `optional_negative_index_latents` | guide latents at t < 0 |
| Motion guides | `optional_guiding_latents` (IC-LoRA) | per-chunk sliced guide latents |

Three structural facts drive almost every tuning decision:

1. **The frozen overlap is the only message between chunks.** Whatever is visible/audible
   in the overlap region propagates; anything off-screen or out of the audio carry is
   *forgotten* and will be re-invented from text when it reappears.
2. **Chunk 0 has no AV context.** It bootstraps everything — sync, identity, motion style —
   from text + image alone. Every later chunk inherits from it. If chunk 0 is wrong, the
   chain faithfully propagates wrong.
3. **Lipsync is cross-modal attention, decided early.** Video motion commits during the
   first (high-sigma) steps. Whatever the audio tokens contain *at those steps* is what
   the mouth can respond to. Free-generated audio is still noise at that point; conditioned
   or carried audio is legible. This asymmetry is behind most sync behavior.

---

## 2. Temporal structure

### `temporal_tile_size`, `temporal_overlap`

Chunk math (pixel frames): chunk 0 covers `tile_size`; every extend chunk adds
`tile_size − temporal_overlap` new frames while re-seeing the last `temporal_overlap`.
At 25 fps with tile 248 / overlap 48: chunk 0 ≈ 10 s, boundaries every 8 s.

- **New-content budget per chunk** = `(tile_size − overlap) / fps` seconds. This is the
  dialog budget: a spoken line must fit inside its chunk's budget (~20 words per 8 s) or
  it gets cut off and the next chunk won't finish it.
- **Overlap is the context window.** ~2 s (48 px) is a validated sweet spot: enough
  evidence of "speech mid-stream / motion mid-arc" without spending the whole tile
  re-generating known content.
- Longer tiles = fewer boundaries (fewer risk points) but more within-chunk freedom for
  drift (e.g. a tracking shot slowly losing its subject); shorter tiles re-anchor more
  often but shrink the dialog budget.
- `video_fps` should be 25 for AV work — audio runs at a fixed 25 latents/second, giving
  1:1 pixel-frame↔audio-frame alignment.

---

## 3. Continuity levers (the sync-critical pair)

### `temporal_overlap_cond_strength` (video) and `audio_overlap_cond_strength` (audio)

These set how strongly each modality's carried-over context is held (mask = `1 − strength`).

**For AV generation, run both at 1.0 — the default.** This is the single most important
setting in the node. At 1.0/1.0 both anchors are clean, immutable, and symmetric — the
regime the model handles as "continue this AV clip." Earlier versions defaulted to
0.5 video / 0.9 audio, inherited from the official *video-only* extend sampler, where a
soft rewritable seam is a blending feature. In joint AV it is a bug: the model can hear
that speech is mid-sentence (audio nearly frozen) but sees the mouth state through 50%
noise and is allowed to repaint it. The modalities then decouple on a per-chunk coin
flip — dialog continues, mouth stops. Check these values when loading old workflows —
saved graphs keep their stored values, not the new defaults.

Asymmetric or partial strengths also cause **boundary teleports**: when a chunk's text
disagrees with where the previous chunk actually ended, a rewritable overlap lets the
model repaint the seam (subject snaps back into frame) instead of animating a correction.

Lower values remain legitimate for **video-only** experiments or when you deliberately
want the seam re-interpreted — but then use `LTXVLoopingSampler` instead.

---

## 4. Text levers

### Global conditioning (guider prompt)

One prompt for every chunk. Best when the scene is *stationary in description*: a locked
character and framing with interpretive room. Techniques that make a global prompt work
across many chunks:

- Describe the background as **soft blur / bokeh / abstract light** — the frozen overlap
  carries actual background continuity, and an abstract description can never contradict
  what emerged.
- License every state the video will pass through: if there are instrumental breaks or
  pauses, sanction them through gaze and stillness ("between lines she holds the viewer's
  gaze"). **Never describe the mouth as closed, shut, settled, or still** — closed-mouth
  language can clamp articulation entirely; the mouth should only ever be mentioned as
  actively speaking.

### `optional_positive_conditionings` (per-chunk script, via MultiPromptProvider)

- Splits on `|`. Mapping is strictly **one prompt per chunk, in order; the last prompt
  repeats** for all remaining chunks. Prompts do not spread across the video.
- To give one story beat N chunks, repeat its segment N times.
- Each segment is encoded independently — repeat the character block and style prefix
  verbatim in every segment.

**Prompt craft rules for chunked generation** (each of these traces to a failure mode):

| Rule | Failure it prevents |
|---|---|
| Identical framing clause opens every segment | crop/zoom re-negotiated at each boundary |
| State framing once, positively — no "never drifts / without ever" | negations plant the failure concept |
| Off-screen things exist only as light, sound, and gaze | naming a destination ("walks toward the door") triggers the view-from-behind prior and breaks framing |
| Segment N+1 opens in the state segment N actually ends in | boundary teleports |
| Motion verbs only where net displacement is wanted; add stopped beats | tracking-shot drift; unstable walk equilibrium |
| Dialog ≤ the chunk's new-content budget | truncated lines the next chunk won't finish |
| With conditioned audio, describe delivery, never quote lines | text fighting the actual track |
| Quoted dialog leaks into *visual* conditioning | a line saying "walking toward it" moves the camera |

### The chunk-0 anchor beat

Because chunk 0 bootstraps sync from nothing, make its segment the most in-distribution
talking configuration available: **static camera, speaker already speaking as the shot
opens, quote immediately after the framing clause, short line (≤ ~20 words) that finishes
with margin.** Locomotion and scene transitions start in segment 2. Once chunk 0 locks,
the frozen context propagates sync for free. (With `audio_cond_strength 1.0` this matters
much less — every chunk, including 0, hears clean audio from the first step.)

---

## 5. Image levers

### `optional_cond_images` + `optional_cond_image_indices` + `cond_image_strength`

- **Index 0 = I2V pin**: encoded directly into the first latent frame. The prompt's
  opening state must *match this image* (pose, framing, setting) or the video opens with
  a fight between text and pin.
- **Other indices = keyframes**: routed automatically to whichever chunk owns that pixel
  frame, added as guide tokens. Use for mid-video waypoints ("by 40 s she's at the door")
  and for the transition anchor when extending existing footage (see `LTX AV Extend
  Latent`, which outputs the right index).
- Negative indices count from the end (`"0,-1"` = first and last frame).

### `optional_negative_index_latents` + `optional_negative_index_strength`

Reference latents placed *before t = 0*, re-applied in **every** chunk. Two things to
understand before using them:

1. **Base-model semantics are "the footage that preceded t = 0"** — not an unordered
   identity reference. This is how official extension passes the previous clip's tail.
   A single full-body reference reads as a benign "a moment ago she looked like this."
   A multi-angle stack reads as *angle-teleporting recent footage*: the last frame in the
   stack acts as a quasi start-frame the opening wants to flow from. If stacking, put the
   angle nearest the opening view last, and keep strength moderate. True unordered
   multi-reference conditioning requires a trained convention (Ingredients-style IC-LoRA).
2. **Why every chunk:** the overlap chain only transmits visible content. A reference
   applied only to chunk 0 is gone by the time an off-screen body part re-enters frame.
   Per-chunk application is what solves push-in/pull-back re-entry drift.

Practical: encode references at exactly the generation resolution; make the subject fill
the frame (reference tokens spent on empty background are wasted); start strength around
0.6 and lower it if chunks open with a pull toward the reference's pose or lighting.

**Division of labor with a character LoRA:** the LoRA carries identity (who she is,
everywhere in its training distribution); a reference pins *instance state* — this exact
armor configuration, these trace routings — details the LoRA treats as free variables.
They stack.

---

## 6. Audio levers

### `audio_cond_strength`

Selects the generation regime for audio:

| Value | Regime | Behavior |
|---|---|---|
| 0.0 | Free joint generation | Audio and video co-emerge. Sync depends entirely on context quality (overlap strengths, chunk-0 anchor). |
| 1.0 | Dubbing | Input audio is clean and audible at every sigma in every chunk. Strongest sync regime; chunk 0 needs no special handling. For pre-made tracks: TTS dialog, songs, scored speeches. |
| 0.5–0.85 | Guided regeneration | Audio content is legible early (video can hear it) but retains `1 − s` freedom to flex toward the video. Output audio will drift from the input — fine for model-generated material, wrong for word-exact TTS. Too low invites boundary discontinuities (each chunk's carry comes from the drifted accumulator, its new region from the pristine input). |

**Track length:** if the input audio ends before the timeline, the remainder is frozen to
zero latents — enforced silence rather than hallucinated continuation (deliberate).
Zero latents are only *pseudo*-silence (may decode to a faint floor, and are
out-of-distribution for the audio→video attention). Prefer matching video length to the
track; if you want a silent tail on purpose, padding the track with real encoded silence
is cleaner than relying on zeros.

### Voice identity across separate generations

`LTX Add Audio Latent Guide` injects reference audio as `ref_audio` tokens (clean,
pre-t = 0) at the conditioning level — the audio analog of image reference conditioning.
Use when separate videos of the same character should share a voice without conditioning
on a full track.

For ID-LoRA voice locking, use the core `LTXVReferenceAudio` node instead (~5s clean
reference clip): it sets `ref_audio` and patches the model with an identity-guidance
pass (extra forward pass per step — wire into stage 1 only; restrict via
`start/end_percent` if compute matters). ID-LoRA checkpoints expect the structured
prompt format `[VISUAL]: … [SPEECH]: exact words [SOUNDS]: voice style + ambient`;
keep the `[SOUNDS]` voice descriptor identical across segments. The sampler carries
`ref_audio` onto per-chunk prompts automatically.

---

## 7. Motion guidance (IC-LoRA)

### `optional_guiding_latents`, `guiding_strength`, `guiding_start_step`, `guiding_end_step`

A full-length control latent (pose tracks, depth, v2v source) sliced to each chunk's
window automatically — the one input that spans the whole timeline besides input audio.
Requires the matching IC-LoRA loaded; on the base model guide latents are weak.

`guiding_start_step` / `guiding_end_step` window the guidance within the sigma schedule:
guidance during early steps controls structure/motion; releasing it before the final
steps lets textures diverge from the guide. The sampler splits the schedule and runs the
segments back-to-back.

---

## 8. Statistics and drift

### `adain_factor` + `optional_normalizing_latents`

Corrects autoregressive *statistics* drift (tone, exposure, saturation pumping over many
chunks). Two modes:

- **No normalizing latents:** every chunk is pulled toward chunk 0's global per-channel
  statistics. Only appropriate when the whole video should share one look — it will fight
  scripted lighting changes (amber corridor → white tunnel → blue deck).
- **With normalizing latents (per-frame):** each frame matches the statistics of the
  corresponding reference frame. The intended two-stage use: feed stage 1's output as the
  normalizing latents in stage 2 — drift correction that respects scripted lighting.

AdaIN fixes statistics only. Identity drift, detail loss, and structural changes are
reference/LoRA problems, not AdaIN problems.

---

## 9. Spatial tiling

`horizontal_tiles` / `vertical_tiles` / `spatial_overlap` trade VRAM for seams: each
spatial tile runs the full temporal pipeline and results are feather-blended. Audio is
taken from tile (0,0) only. Prefer resolution staging (generate lower, upsample, refine)
over spatial tiling when possible — per-tile prompts don't exist, so tiles see the whole
scene description.

---

## 10. Two-stage workflows

Stage 1 (full generation at lower resolution) decides **everything that moves**: sync,
motion, framing, pacing. Stage 2 (latent-upsampled, denoise ≈ 0.3, `audio_cond_strength
1.0` against stage 1's audio) polishes texture and detail. Consequences:

- Every sync/framing/motion problem is a **stage-1 problem**. Do not try to fix them with
  stage-2 settings; stage 2 cannot reopen a mouth or undo a teleport.
- Stage 2's input latent carries the stage-1 prior into every chunk's initialization —
  its chunks are anchored by a full-length source, so its overlap strengths matter far
  less than stage 1's.
- Use the tiled AV upsampler between stages only when feeding a refinement pass (the
  refine smooths tile statistics); use the non-tiled one otherwise.

---

## 11. Seeds

Per-chunk seeds derive from the base seed + chunk position, so a re-run with the same
seed reproduces every chunk. The hidden `per_tile_seed_offsets` parameter (comma-separated
ints, indexed by temporal chunk) can re-roll a single chunk without touching the others —
e.g. `"0,0,7"` re-rolls only chunk 2. Currently code-level only (not exposed as a widget);
expose it if surgical re-rolls become routine.

---

## 12. Recipes

**Long-form talking head, free dialog (the CyberNoeve pipeline)**
Overlap strengths 1.0/1.0 · tile ~10 s, overlap ~2 s · per-chunk script with anchor beat
first · dialog ≤ budget per segment · character LoRA · stage 2 at denoise ~0.34 with
stage-1 audio at 1.0.

**Music video from a track (T+A2V)**
`audio_cond_strength 1.0` · video length = track length · global prompt: locked character
block, bokeh background, pauses licensed via gaze/stillness (no closed-mouth language) ·
no quoted lyrics.

**Image + speech dubbing (I2V+A2V)**
Cond image at index 0 · prompt opening state matches the image exactly, "already speaking
as the shot opens" · `audio_cond_strength 1.0` · camera motion carried by the prompt only.

**Extend existing footage**
`LTX AV Extend Latent` builds the input latent and mask · its `extension_start_frame` →
`optional_cond_image_indices` with the last existing frame as the keyframe image ·
`existing_denoise` 0.1–0.3 for light seam refinement.

**Character state pinning across camera moves**
Character LoRA for identity + single full-body reference in
`optional_negative_index_latents` at strength ~0.6 · reference at generation resolution,
subject filling frame.

**Surgical fix of one bad chunk**
Same base seed, `per_tile_seed_offsets` with a nonzero entry at the offending chunk index.

---

## 13. Symptom → lever

| Symptom | First lever |
|---|---|
| Lips decouple from dialog in some segments | overlap strengths → 1.0/1.0 |
| Weak/soft lipsync in chunk 0 only | anchor-beat segment 1, or condition the audio |
| Subject snaps/teleports at ~boundary times | prompt N+1 assumes a state chunk N didn't reach; overlap strength |
| Subject slowly drifts out of frame | locomotion in prompt; add stopped beats, slow pace |
| Camera breaks framing to show something | an off-screen object is named visually — convert to light/sound/gaze |
| Zoom/crop pumping between segments | framing clause not identical across segments |
| Audio stutter exactly at boundary multiples | stitch bug (fixed 2026-07-11) — update the node |
| Progressive AV desync late in video | same stitch bug; verify against a conditioned track |
| Tone/exposure drift over minutes | `adain_factor` (per-frame mode in stage 2) |
| Body parts change after leaving and re-entering frame | negative-index reference (every chunk), LoRA for identity |
| Chunks open pulled toward the reference's pose | lower `optional_negative_index_strength`; stack order |
| Faint hum after the input track ends | zeros-tail pseudo-silence — match lengths or pad with encoded silence |
| Mouth stays clamped shut during speech | closed-mouth language in the prompt ("lips settle closed", "mouth still") — remove; license pauses via gaze only |
