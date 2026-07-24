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

### `temporal_tile_size`, `temporal_overlap`, `scene_lengths`

Chunk math (pixel frames): **every chunk — including chunk 0 — delivers one
`tile_size` of new content**; extend chunks additionally re-see the last
`temporal_overlap` as frozen context (their sampled window is `overlap + tile`).
At 25 fps with tile 248 / overlap 48: one prompt segment ≈ 10 s of footage,
uniformly. The loop walks this schedule exactly and stops when the timeline is
covered (the last chunk clamps to the remainder — size totals as multiples of
the tile to avoid a short tail beat).

`scene_lengths` (pipe/comma-separated pixel-frame counts, multiples of 8)
replaces the uniform tile with per-chunk scene lengths — a 4 s punchy beat and
a 14 s monologue beat in one run, prompts mapping 1:1 to scenes. Use the
**LTX Scene Length Calculator** to author scenes in seconds; it emits the
snapped `scene_lengths` string *and* the exactly-matching `frame_count` for the
empty latent, so the schedule and the canvas cannot disagree.

- **New-content budget per chunk** = `tile_size / fps` seconds (or that scene's
  length). This is the dialog budget: a spoken line must fit inside its chunk's
  budget or it gets cut off and the next chunk won't finish it.
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

## 5b. Spatial denoise mask (inpainting)

### `optional_denoise_mask`

A spatial mask over the generation: **white = regenerate, black = keep** pinned to the
input latent's video content (standard Comfy denoise-mask polarity). This is the pack's
**primary inpainting path — no inpaint LoRA required.**

Why no LoRA: the LTX inpaint / in-outpainting IC-LoRAs exist only to teach the model
"this painted-color region means synthesize here, reproduce the rest." The denoise mask
states that *structurally*, so the LoRA's role on the **where** is redundant. And the fill
coheres with the kept region — matching its light, shadow, and geometry — because the model
sees the pinned latent **while** denoising: it completes one scene, not two glued layers
(the difference from a decode-then-composite).

**Wiring (base-model inpaint):**

```
source video → VAE encode (video) ─┐
source/empty audio → (audio) ───────┴─ LTXVConcatAVLatent → latents   (the pinned content)
SAM / painted mask ───────────────────────────────────────→ optional_denoise_mask
```

- No guide, no color fill, no IC-LoRA. Just the source in the init latent and the mask.
- **Requires real video in the input AV latent** — kept regions reproduce it, so an
  all-zeros latent pins **black** (the sampler warns). This is a v2v tool.
- Prompt describes the *whole* scene as it should look after the edit (the masked region
  included), same as any inpaint prompt.
- `video_cond_strength` stays **0** — the mask does the holding. (A nonzero value would
  also hold the white region: the merge is elementwise-min, so `min(1, 1−vcs) = 1−vcs`
  leaks a hold into the regen area.)

**Mask formats:** a single mask is static across all frames; a mask **batch** is resampled
onto the latent grid (per-pixel-frame masks — e.g. SAM at the video's frame count — land
directly; per-latent-frame batches also accepted). The mask min-merges keep-wins with the
`video_cond_strength` / overlap / keyframe masks, so a keep region always wins. Spatial
tiling is supported (unlike small-grid guides). The console prints `denoise mask active:
N% kept` — sanity-check it against what you painted (≈0% kept = inverted polarity).

**Boundary artifacts:** if halos appear where kept meets regenerated, feather the mask
(soft grey falloff) — fractional values interpolate through the same merge math.

### When the IC-LoRA route still wins (the fallback)

Reach for **LTX Inpaint Color Fill** + an inpaint IC-LoRA (guide + `guiding_downscale_factor`
from metadata, §7) for the cases the base model struggles with: **large holes** where most
of the region must be hallucinated from little surrounding context, and **hard semantic
edits** the base model resists. Color Fill paints the masked region the LoRA's trained
color (magenta / chroma green / Lightricks green) — composite at final resolution so the
boundary stays exact. For ordinary object removal, face swaps, and localized changes,
base + mask is simpler and sufficient.

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

**The only working voice lock is the ID-LoRA path**: core `LTXVReferenceAudio`
(~5s clean reference clip) with the TalkVid ID-LoRA loaded. It sets `ref_audio`
and patches the model with an identity-guidance pass (extra forward per step —
wire into stage 1 only; restrict via `start/end_percent` if compute matters).
ID-LoRA checkpoints expect the structured prompt format `[VISUAL]: … [SPEECH]:
exact words [SOUNDS]: voice style + ambient`; keep the `[SOUNDS]` voice
descriptor identical across segments. The sampler carries `ref_audio` onto
per-chunk prompts automatically. LoRA loading (KJ per-block loader): the
voice lives in the audio-OUTPUT groups — `audio` + `video_to_audio` (+ `other`)
at 1.0; `video` and `audio_to_video` can be zeroed for a visuals-untouched lock.

**What does NOT work (all empirically dead, 2026-07):** the base model does not
adopt voice identity from context — three mechanisms tested and failed:
`LTX Add Audio Latent Guide` without the ID-LoRA (the `ref_audio`
negative-coordinate placement is an ID-LoRA training convention; the node is
marked ARTIFACT), carry-swap reference audio (with and without v2a severed —
the ref transfers only via forced utterance-continuation, never as identity).
Voice consistency within a generation is content lineage, not timbre adoption;
across generations, only the trained pathway transfers a voice.

### Multi-speaker dialog (turn-based)

> **⚠️ Status (2026-07): multi-voice is ABANDONED as non-functional** — the
> ID-LoRA's reference is globally amplified and inherently single-voice; per-chunk
> switching degraded in testing (garbled onsets, wrong-speaker bleed). HOWEVER:
> that testing predates the keep-carry-verbatim stitch and the turn-seam
> choreography rules, both of which independently attack the observed failure
> modes — a retest under current conditions is pending and the machinery below
> remains installed for it. Mixed-gender pairs are the most promising case
> (gender is a hard prior; text switches the voice categorically). Single-ref
> use of the Multi node is a working drop-in for the core node.

`LTX AV Reference Audio Multi (ID-LoRA)` + `LTX AV Speaker Prompt Provider` extend
ID-LoRA voice locking to up to four speakers, one per chunk. In the script, write
`[SPEAKER n]:` in place of `[SPEECH]:` — the provider records the routing and rewrites
the tag to `[SPEECH]:` before encoding, so every chunk is individually in-convention
for the LoRA; the multi-speaker structure exists only in the scheduler. Rules:

- **One speaker per chunk.** Overlapping/simultaneous speech needs trained per-token
  binding (MultiTalk-class) and is out of scope.
- **`[SPEECH]` content must be in spoken register.** The alignment was trained on
  people talking; written-register prose (definitions, gerund chains, nominalizations)
  garbles as repeated/rearranged words regardless of speaker. Read the line aloud —
  if it's stiff to say, it will garble. Keep lines inside the chunk's budget
  (~2.5 words/sec) with margin. **Punctuation: periods, commas, and question marks
  only — never dashes** (typographic, not phonetic; speech parsing mishandles them).
- **`[SOUNDS]` is one flat declarative sentence about the speaker**, matching the
  trained register: `The speaker has a clear, assured female voice with a
  conversational tone.` Identical across that speaker's turns. No fragment chains,
  no ambient-event lists, no temporal narration — stylized clauses like "then a beat
  of quiet" are off-register and their words can leak into the spoken audio
  (observed: a run spoke the word "beat").
- **`[VISUAL]` must bind the voice to the right face**: both character blocks appear
  verbatim in every segment; the speaker gets the articulation clause; the listener is
  described through gaze and stillness only (never mouth state).
- **Turn-boundary containment**: if a turn misbehaves, the frozen context faithfully
  propagates the damage ~2 chunks before self-recovery. The tool is
  `per_tile_seed_offsets` (`0,0,7` = re-roll chunk 2 only) — re-roll the originating
  chunk. Do **not** attempt to loosen the audio carry at boundaries: audio freedom in
  a region where video is frozen lets the model start the new line under the previous
  speaker's face (tried and removed).
- Untagged `[SPEECH]:` segments fall back to speaker 1 — single-voice scripts run
  unmodified.

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

### Small-grid IC references (`guiding_downscale_factor`)

Some IC-LoRAs are trained on a **reduced reference grid** — their safetensors
metadata declares `reference_downscale_factor` (pixel spatial upscaler x2 = 2,
x4 = 4; read it with `LTX LoRA Metadata Reader`, don't trust example-workflow
widgets). For these, feed the guiding latent at `gen_dims/factor` — for a 2×
upscale pass that is the **stage-1 latent directly, no resizing** — and set
`guiding_downscale_factor` (wire it from the Metadata Reader so it can't drift).
Each chunk's guide slice is dilated onto the full grid (holes filtered before
attention, so overhead is only the small-grid token count) with RoPE spans
covering factor-larger patches — the trained geometry. A dense full-res
reference leaves these LoRAs **silently inert**; the wrong factor errors with
expected-vs-got dims. Factor 1 = normal dense references (Ingredients,
CrossView, depth/pose). Not compatible with spatial tiling >1×1 yet.

**Conditioning hygiene (all IC use):** feed the sampler clean text encodes only.
Conditioning that passed through an official guide node upstream carries
full-timeline `keyframe_idxs`/attention entries that cannot survive chunk
remapping — guaranteed bookkeeping crash. References enter via
`optional_guiding_latents`, never via pre-built guide conditioning.

## 7b. Dual-sampler schedules (phase 2)

`optional_phase2_sampler` + `optional_phase2_guider` + `phase2_start_step` reproduce
the dual-sampler pattern (heavy solver for structure, cheap solver for refinement)
inside every chunk. Example, porting a Clownshark KSampler → ChainSampler pair:

- Your custom schedule (e.g. Linear Quadratic Advanced) → `sigmas`
- Phase 1: ClownSampler (etdrk2_2s, eta 0.5, options) → `sampler`; CFGGuider @2.0 → `guider`
- Phase 2: ClownSampler (euler, eta 0) → `optional_phase2_sampler`; CFGGuider @1.0 → `optional_phase2_guider`
- `steps_to_run 4` → `phase2_start_step = 4`

The handoff is resample-style continuation (re-noise the current estimate at the
segment's first sigma), matching ChainSampler's `resample` mode. The phase-2
guider's own conditioning is discarded — it inherits the chunk's full conditioning
(per-chunk prompt, guides, keyframes, ref_audio); only its guidance settings (cfg)
apply. Near-facsimile, not bit-exact: solver-internal state does not cross the
phase boundary.

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

### Pixel-upscale stage 2 (IC-LoRA, chunked)

The generative alternative to the latent-upsampler refine — a **full generation
from noise** at target res with stage 1 as small-grid in-context reference
(§7). Wiring: empty AV latent at target res (stage-1 audio concatenated,
`audio_cond_strength 1.0`); stage-1 video latent → `optional_guiding_latents`
directly; same `scene_lengths` and per-chunk prompts as stage 1 (seams align →
each chunk references its own stage-1 content); upscaler LoRA loaded with
factor wired from metadata; `guiding_strength 1.0` (loosen adherence via
`guiding_end_step`, never sub-1.0 strength); full-gen sigma schedule, not a
low-sigma tail. Stage 1 can drop to very low res (~quarter / ~280p — the
3-stage dimension calculator's ÷128 constraint is exactly the x4 divisibility
requirement): structure and AV sync commit there, detail is synthesized here.

## 11. Seeds

Per-chunk seeds derive from the base seed + chunk position, so a re-run with the same
seed reproduces every chunk. `per_tile_seed_offsets` (widget; comma-separated ints,
indexed by temporal chunk) re-rolls a single chunk without touching the others —
e.g. `"0,0,7"` re-rolls only chunk 2. The surgical fix for one bad chunk in an
otherwise good run.

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

**Inpaint / object removal (base model, no LoRA)**
Source video → encode → `latents` (the pinned content) · SAM/painted mask →
`optional_denoise_mask` (white = the region to change) · `video_cond_strength 0` · prompt
describes the finished scene · no guide, no color fill. IC-LoRA route (Color Fill + inpaint
LoRA) only for large holes / hard edits (§5b).

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
| No movement — video holds the still (same image used as all keyframes) | `cond_image_strength` → ~0.5 — identical hard anchors at every boundary make stasis the optimal trajectory (staying on the image satisfies every constraint); at 0.5 they act as look-lock attractors and motion returns. Distinct travel keyframes keep 1.0 (arrival is the point) |
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
| Speech garbled: words present but repeated/rearranged | written-register `[SPEECH]` line — rewrite in spoken register (read it aloud) |
| Stray word spoken at a turn's end, next 1–2 chunks garbled | `[SOUNDS]` vocabulary leak + carry contagion — flatten `[SOUNDS]` to the declarative register; re-roll the origin chunk via `per_tile_seed_offsets` |
| Seam "speaks" the next chunk's prompt / re-voiced boundary | keep-carry-verbatim stitch (fixed 2026-07-15) — runs automatically at `audio_overlap_cond_strength ≥ 1.0`; the frozen carry is no longer regenerated under the next chunk's conditioning. Also kills static-image burn-in and continuation gibberish. |
| Gibberish generated speech over a guide video (changing the words) | the guide's visible lips out-muscle the text via v2a — set `v2a_cross_attn = False` on the **LTX AV Cross-Attention Toggle** node. Text drives the audio; a2v resyncs the mouth. Only bites during generation; irrelevant with input audio. |
| Lips move during silence over a guide video | the guide's original lip motion bleeds through where the new audio is silent — fill the runtime with dialog (no dead air), and/or free the mouth more (higher CrossView regeneration, lower guide/overlap strength) so a2v dominates it. |
| Crash: `expanded size of tensor … must match` inside attention, with MultimodalGuider | STG perturbed pass × any sub-1.0 guide strength (incl. `cond_image_strength 0.5`) breaks core's guide-mask attention. Set `perturb_attn=false`/`stg=0` in GuiderParameters (keeps per-modality CFG), or keep all guide strengths at 1.0 and window via `guiding_start/end_step` |
| Crash: `guide pre_filter_counts != keyframe grid mask length` | stale guide conditioning — either an upstream guide node (LTXVAddGuide / IC-LoRA guide) or a cached cond polluted across queue runs. The sampler now **auto-strips** guide bookkeeping on entry (prints `stripping stale guide conditioning …`) and no longer memoizes onto the cached guider, so this should be fixed. If it still fires, note what the strip message names and report it; references go through `optional_guiding_latents` only |
| Inpaint: masked region unchanged, or whole frame drifts | `optional_denoise_mask` — unchanged = polarity inverted (white must be the regen region; check the `N% kept` console line) or the mask isn't reaching the model; whole-frame drift = mask absent so nothing is pinned. Needs real video in the input latent (§5b) |
| Inpaint fill looks pasted-on / large hole invents wrong structure | base model has too little surrounding context — switch to the IC-LoRA route (LTX Inpaint Color Fill + inpaint LoRA, §5b fallback) |
| Crash: `size of tensor a … must match b` in `_linear_overlap_blend` with a prior latent | the continuation prior is shorter than `temporal_overlap` — use a prior of at least the overlap in pixel frames (≥ ~2s at overlap 48; more gives the carry real substance) |
| Keyframes silently ignored (unconditioned output despite indices set) | `optional_cond_images` missing/bypassed — indices without images degrade silently by design. Also check count: images pair with indices via zip, and the shorter list wins (extra indices dropped without warning; Keyframe Planner's `count` output is the check) |
| Small-grid IC-LoRA (pixel upscaler) has no effect | reference fed dense/full-res — these LoRAs need `guiding_downscale_factor` set (from metadata) with the guide at `gen_dims/factor` (§7 small-grid) |
