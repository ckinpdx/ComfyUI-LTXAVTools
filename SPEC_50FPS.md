# SPEC: 50 fps Long-Form AV Support

Status: **IMPLEMENTED 2026-07-27.** Validated: `delta 0` from LTX AV Latent
Check at 25 fps (regression — counts unchanged from the pre-refactor build) and
at 50 fps with `audio_cond_strength = 1.0`, plus multi-segment runs at shorter
durations in normal use. §6.4 (60 s cumulative drift) and §6.5 (free-gen lipsync
battery at 50) are **parked, not failed** — every boundary in a multi-segment run
exercises the same telescoping identity a 60 s run would, just fewer times, so
the long test now confirms rather than discovers. Worth running before anyone
leans on 50 fps for a long single take.

Review notes folded in before implementation: the core identity is verified
algebraically (§3.1 closed form), `a_carry` collapses into `audio_pos` (§3.2),
the `s ≥ ov` invariant is explicit (§3.2.1), and the scope table covers the
three audio features added after this spec was first written (§4).

### Findings from implementation (not in the original spec)

1. **`frame_rate` lives on the CONDITIONING, and its absence is silent.**
   `LTXVConditioning` is the only thing that sets it (`nodes_lt.py` →
   `conditioning_set_values(..., {"frame_rate": ...})`); `model_base.py` reads
   it **per conditioning** with `kwargs.get("frame_rate", 25)`. A graph with no
   `LTXVConditioning` node runs the model's temporal RoPE at **25 regardless of
   the sampler widget** — the sampler's `video_fps` governs audio arithmetic
   only. Symptom: correct-by-its-own-math audio against 25 fps-paced video
   (`delta 0`, but the result only looks right at 25 fps playback).
2. **MultiPromptProvider drops `frame_rate`** — it encodes bare text, so the
   per-chunk conds have no such key and default to 25, while the negative
   (which came through `LTXVConditioning`) keeps the real rate: **the two CFG
   branches on different time axes.** Fixed by a carryover in
   `_prepare_guider`, mirroring the `ref_audio` fix. Invisible at 25 fps
   because the default coincides with the rate.
3. **Off-grid fps no longer accumulates.** The spec framed non-divisor rates as
   merely unsupported; in fact deriving every length from one global map means
   the total telescopes to `audio_pos(total)` at **any** fps. What is lost
   off-grid (24, 30) is only per-boundary position quantization (≤ ~20 ms),
   which does not compound. 24 fps went from broken to workable.
4. **The temporal upsampler IS the 25 → 50 path, and audio needs no change.**
   `audio_pos(T, 25) == audio_pos(2T−1, 50)` exactly (q halves as the latent
   count doubles), so a temporally-upsampled 25 fps AV latent is already a
   valid 50 fps one — verified for T = 5, 10, 19, 40, 97. This is preferable to
   native-50 generation: it keeps the entire sync/prompt doctrine that was
   tuned at 25. Everything downstream must then be told 50 (Latent Check,
   Streaming Save, any second-pass `LTXVConditioning`).

---

## 1. Problem

Audio runs at a fixed **25 latents/second** regardless of video fps. The
sampler currently computes all audio counts from **chunk-local pixel spans**:

```python
T_a       = round(((T_v - 1) * 8 + 1) / fps * 25)     # per chunk
a_overlap = round((1 + (ov - 1) * 8)  / fps * 25)     # per carry
```

At 25 fps, pixels ↔ audio latents are 1:1 and every count is integer-exact.
At 50 fps, every LTX pixel count is odd (`8n + 1`), so **every one of these
expressions lands exactly on x.5** and gets rounded — with Python's
round-half-to-even flipping direction by parity. The stitch identity
`T_a_chunk − a_overlap = num_new × (200/fps)` then breaks by ±1 audio frame
(±20 ms) at boundaries, parity-dependent, cumulative. Same failure family as
the fixed drop-1/pad-1 stitch bug, reintroduced as rounding dust.

Single-shot 50 fps AV (one window, no chunking) is unaffected — this is purely
a chunk-boundary accounting problem.

## 2. Key insight: boundaries are exact, local spans are not

Global chunk boundaries always sit at whole video latents. The pixel span of
any run of latents `[m, n)` **not touching position 0** is `(n − m) × 8` —
even — and its audio equivalent at 50 fps is `(n − m) × 4`, an integer. The
odd pixel counts (and the half-frames) come exclusively from the **first-frame
asymmetry** (latent 0 = 1 pixel), which every chunk-local window re-introduces
at its local position 0.

Therefore: **derive all audio counts as differences of one global boundary
map**, and the roundings telescope away — only chunk 0's single half-frame
remains, absorbed once at the timeline start.

## 3. Design

### 3.1 The global audio boundary map

```python
def audio_pos(n_latents: int, fps: float) -> int:
    """Audio latents covering video latents [0, n). Half-up rounding —
    NEVER Python round() (banker's rounding is parity-dependent)."""
    if n_latents <= 0:
        return 0
    px = (n_latents - 1) * 8 + 1
    return int(px * 25.0 / fps + 0.5)
```

Properties (for `q = 200 / fps` integer, e.g. q=8 @25fps, q=4 @50fps):

**Closed form.** For `n ≥ 1` the expression reduces to

```
audio_pos(n) = (n − 1)·q + floor(q/8 + 0.5)
```

because `(n − 1)·q` is an integer whenever `q` is, so it factors straight out of
the rounding. Everything below follows from this:

- `audio_pos(n) − audio_pos(m) = (n − m) × q` **exactly**, for all `n > m ≥ 1`
  — the rounding term is a *constant* and cancels.
- Only `audio_pos(n) − audio_pos(0)` carries a rounding (+0.5 → half-up), once.
- Telescoping: summing per-chunk contributions reproduces `audio_pos(total)`
  exactly — cumulative drift is impossible by construction.

**Float safety (do not "improve" this).** `int(px * 25.0 / fps + 0.5)` is exact
in float64 for every valid fps: `fps | 200 = 2³·5²` forces `fps = 2^a·5^b`, so
`25 / fps = 5^(2−b) / 2^a` — a dyadic rational, exactly representable. No
`Fraction` / `Decimal` needed, and switching to them would only add cost. This
guarantee holds *only* for the exact-fps set; it is another reason to reject
other rates rather than approximate them.

### 3.2 Per-chunk quantities

For extend chunk with global new-content span `[s, e)` (latents) and overlap
`ov`:

| Quantity | Formula | @50 fps |
|---|---|---|
| New audio (kept) | `a_new = audio_pos(e) − audio_pos(s)` | `4 × (e − s)`, exact |
| Carry given to model | `a_carry = audio_pos(ov)` | `4·ov − 3` |
| Carry slice from accumulator | `acc[audio_pos(s) − a_carry : audio_pos(s)]` | end-anchored |
| Chunk audio length | `T_a_chunk = a_carry + a_new` | integer |
| Stitch join point | `join = audio_pos(s) − a_carry` | |
| Result | `acc[:join] ++ chunk_audio` → length `audio_pos(e)` | **exact** |
| Input-audio conditioning slice | `audio_full[audio_pos(s) : audio_pos(e)]` | exact |

Chunk 0: `T_a_0 = audio_pos(n_0)`. Final trim: `audio_pos(total_latents)`.

**`a_carry` is `audio_pos(ov)` — not a separate rounding.** The local carry span
`(ov − 1)·8 + 1` half-up-rounded is *identically* `audio_pos(ov)` (`8ov − 7`
@25, `4ov − 3` @50). Use the one function. This matters beyond tidiness: with a
single `audio_pos` everywhere there is no second formula to drift, which is the
regression §7 warns about.

### 3.2.1 Required invariant: `s ≥ ov`

The carry slice start `audio_pos(s) − audio_pos(ov)` is `≥ 0` **iff `s ≥ ov`**
(`audio_pos` is monotonic). This is not decoration — a negative start in Python
does not raise, it silently wraps to `len − k` and yields plausible-looking
garbage audio.

The invariant currently holds because `_sample_extend_chunk` clamps
`temporal_overlap` to the accumulator length (short-prior guard), which forces
`ov ≤ acc_T = s`. **Assert it explicitly** rather than relying on that clamp
staying put:

```python
assert s >= ov, f"carry underflow: s={s} < ov={ov}"   # or a raise with context
```

A future change to the clamp would otherwise break audio silently rather than
loudly.

The carry length stays **locally consistent** (the model sees a window whose
local latent 0 is 1 px, i.e. half an audio latent at 50 fps — `a_carry` is the
half-up rounding of that local span). The **end** of the carry is anchored to
the exact global boundary `audio_pos(s)`; the rounding lives only in where the
carry *starts* — a ≤10 ms positional quantization of context, constant per
chunk, and non-accumulating because every join point and every length is an
`audio_pos` difference.

### 3.3 Residual error budget

- Interior boundaries: **0** length error (telescoping), ≤½ audio latent
  (10 ms) of carry-start quantization in what the model *hears* as context.
  Inaudible; does not move output audio.
- Timeline start: one half-up rounding (chunk 0 audio is ≤10 ms longer than
  the ideal fractional value). One-time.
- At 25 fps all formulas reduce **exactly** to current behavior (q = 8,
  `a_carry = (ov−1)·8+1`) — this refactor is a strict generalization; 25 fps
  outputs are bit-identical.

## 4. Scope of changes

| Site | Change |
|---|---|
| `av_looping_sampler.py` helpers | Replace `_audio_frames_for_video_chunk` / `_audio_overlap_frames` with `audio_pos` differences; thread global `s`/`e` (already available post-scheduler-refactor: `v_start`/`v_end`) into `_sample_first_chunk` / `_sample_extend_chunk` instead of deriving audio counts from local `T_v` |
| Stitch | Join point and lengths from `audio_pos` (formulas above) |
| `_debug_chunk` | Report global audio spans + expected from `audio_pos` |
| **Audio noise mask** | Built from `a_overlap` / `T_a_chunk` — rederive as `a_carry` / `a_carry + a_new` |
| **`optional_ref_audio_bank` (carry-swap)** | Slices `ref[:, :, :a_overlap]` and tiles short refs against it — becomes `a_carry` |
| **`trailing_silence_frames`** | Splices at `a_overlap − n_sil`; clamp `n_sil ≤ a_carry − 1` against the new value |
| `video_fps` widget | Validation: warn when `200 / fps` is not an integer ("exact boundaries only at fps ∈ {1, 2, 4, 5, 8, 10, 20, 25, 40, 50, 100, 200}; expect ≤20 ms boundary quantization otherwise") |
| `audio_latent.py` `LTXAVExtendLatent` | Same `audio_pos` math for `total_audio` / `ext_audio` |
| `utils.py` `LTXAVLatentCheck` / `LTXAVSeparateCheck` | `expected = audio_pos(T_v, fps)` (currently hardcodes the 25 fps 1:1 formula) |
| Docs | Guide §2 + §audio-alignment; README `video_fps` row |

Out of scope / unchanged: video-side chunk math (fps-agnostic), spatial
tiling, upsamplers (audio passes through), audio guide nodes (fps-agnostic
token format), Scene Length Calculator (already fps-parameterized).

## 5. Workflow requirements at 50 fps (user-side)

- `video_fps = 50` on the sampler **and** frame-rate conditioning set to 50
  upstream (`LTXVConditioning`) — the model's AV cross-attention RoPE takes
  the frame rate from conditioning; a mismatch desyncs regardless of sampler
  math.
- Scene Length Calculator / Frame Calculator run with `fps = 50`.
- Model must be a 50-fps-capable AV checkpoint (LTX-2.x native 50).

## 6. Test plan

1. **Regression @25**: fixed-seed run pre/post refactor → outputs bit-identical.
2. **Single-shot 50 baseline**: one-window AV gen at 50 fps outside the looper
   (establishes the model-side quality bar).
3. **Boundary null test @50**: 2-chunk run conditioned on a known track
   (`audio_cond_strength 1.0`) → output audio must align sample-accurately
   with the source across the join; listen + waveform-diff at the boundary.
4. **Drift test @50**: 60 s conditioned run → alignment at t=0 vs t=60 s
   identical (no cumulative offset).
5. **Free-gen lipsync @50**: the standard talking-head battery (anchor beat,
   1.0/1.0 overlaps) — verifies the 25-trained sync behaviors transfer to the
   native-50 regime.

## 7. Effort

Contained: **one** audio-count helper (`audio_pos` — it subsumes both the chunk
and carry formulas, see 3.2), threading two integers (`s`, `e`) through existing
call paths, the three audio-side features added since this spec was written
(noise mask, carry-swap bank, trailing silence), three secondary nodes, docs.
Roughly a day of careful work plus the test battery — the §6 battery is more,
mostly the 60 s drift run and re-validating the sync doctrine at 50.

Highest-risk items, in order:

1. **Carry-start anchoring (3.2).** Implement exactly as specified. The
   temptation to "simplify" it back to a chunk-local formula is how the .5s
   return. Keeping `a_carry = audio_pos(ov)` removes the second formula that
   would drift.
2. **The `s ≥ ov` invariant (3.2.1).** Assert it. Violation is silent, not loud.
3. **Regression @25 must be bit-identical** (§6.1) before anything at 50 is
   trusted — the reduction is proven algebraically above, so any diff means an
   implementation slip, not a design problem.
