# SPEC: Native IC-LoRA Reference Support (Ingredients-class)

Status: **v1 BUILT AND WORKING (2026-07-27).** Confirmed live by the user:
references bind through the looping sampler. Because the pack is inert on a
stock sampler by design, "it works" necessarily means the real path ran —
per-chunk carryover, injection, crop and all.

§4.1 convention diff PASSES exactly (1/3/4 views x factor 1/2/4, maxdiff 0 on
latent, noise_mask, keyframe_idxs start+end, and the attention entry).

**MSR end-to-end is verified in use** (2026-07-27) — the `as_sequence` layout
feeding the Licon MSR reference video through the pack, per chunk, works.

Node: `nodes/ic_reference.py`; sampler hook: `_prepare_guider` carryover + one
call in each chunk builder.

Remaining items are SCOPE, not verification: single spatial tile only (§2.4),
one pack per run (multi-pack is the Multi IC-LoRA extension).

Previously parked 2026-07-13: the motivating LoRA (`LTX-2.3-22b-IC-LoRA-Ingredients`) turned out to use *control
geometry* — a static composite-sheet video as v2v reference latents at
downscale 1 — which the existing `optional_guiding_latents` input already
serves (repeat the sheet image to full length, encode, feed; prompt as
`Reference sheet: … Generated video: …`). Nothing needed the out-of-band path.

**Un-parked because MSR needs exactly this.** The Licon MSR LoRA
(`LTX-2.3-Multiple-Subject-Reference`) conditions on a latent-aligned reference
*video* — subjects on whole latent frames, background last — that has no
timeline correspondence and must therefore be injected whole into every chunk.
That is precisely the out-of-band geometry this spec describes, and it is the
live consumer. Multi IC-LoRA's video half is the same shape.

**JoyEcho is NOT a consumer and is closed (2026-07-27, decision).** Its
mechanism was studied and is not portable: it requires
`echo-longvideo-release.safetensors`, a 46 GB LTX-2.3-*derived* fine-tune
(~46–50 GB peak VRAM, no stated LoRA support, non-commercial licence). The
memory behaviour lives in retrained weights, not in the inference patches.
Do not re-open this as an implementation target.

What made the build small: **the per-chunk append routine already existed.**
`_add_latent_guide()` in `av_looping_sampler.py` already mirrors the official
convention end-to-end — dilation at `downscale_factor > 1`, `append_keyframe`,
and the *measured* `pre_filter_count` (conditioning delta rather than a
predicted formula). It is in production service for the small-grid pixel-upscale
IC-LoRA. §2.1's `apply_ic_references()` is therefore a thin caller, not a
reimplementation — which retired most of the §5 risk.

---

## 1. Problem

The sampler supports two conditioning geometries today:

- **Temporal control** (`optional_guiding_latents`): a full-length control
  track sliced per chunk — depth/pose/canny/motion-track/v2v. Frame t of the
  guide corresponds to frame t of the output.
- **Timeline anchors** (`optional_cond_images`, overlap carries): content
  pinned at specific timeline positions.

Reference-type IC-LoRAs (Ingredients, 2.3-compatible) are neither: a static
set of identity views that must condition **every chunk, out-of-band**, with
no timeline correspondence.

The official `LTXAddVideoICLoRAGuide` cannot be used upstream: it appends the
encoded reference frames **into the latent tensor** and writes matching
`keyframe_idxs` into conditioning — a latent+conditioning pair. The looping
sampler rebuilds its latents per chunk, so the reference frames never enter
any chunk while the conditioning still promises them. Observed failure:
maskless T2V chunk + external `keyframe_idxs` → unguarded
`patchify(denoise_mask=None)` crash in core (now defended by the always-mask
fix, 2026-07-13); with a mask it would be silent corruption instead.

## 2. Design

Upstream-node architecture (the Reference Audio Multi pattern): configuration
and machinery live in a dedicated node; the payload rides the conditioning
wire as inert metadata; the sampler contributes a small per-chunk hook. Zero
new sampler inputs — no widget-order exposure, no graph changes.

### 2.1 New node: `LTX AV Add IC References` (own module, e.g. `nodes/ic_reference.py`)

**As built (2026-07-27)** — deviates from the original draft in two places,
both noted below:

| Input | Type | Notes |
|---|---|---|
| `positive` / `negative` | CONDITIONING | passthrough; pack attached to positive |
| `vae` | VAE | reference encoding |
| `reference_images` | IMAGE | batch of views (e.g. 4 multiview stills) |
| `width` / `height` | INT | **generation** pixel dims. *Draft had no such input;* the node needs the target grid to size references and there is no latent on the wire to read it from. Validated against the chunk at inject time — mismatch is a hard error, not a silent stretch |
| `strength` | FLOAT, default 1.0 | IC rule: 1.0; below 1.0 known to bleed |
| `latent_downscale_factor` | FLOAT, default 1.0 | *Draft took a `MODEL` and read metadata off it.* But `LTXICLoRALoaderModelOnly` **returns the factor as a FLOAT output** rather than attaching it to the model, so a MODEL input would have had nothing to read. Taking the float directly also accepts `LTX LoRA Metadata Reader`, and matches how the sampler's own `guiding_downscale_factor` is already wired |
| `layout` | COMBO | `one_latent_per_view` \| `as_sequence` \| `single_frame`. *Not in the draft* — see §2.3.1: it defuses a silent data-loss trap, and `as_sequence` is what MSR requires |
| `crop` | COMBO | `disabled` \| `center` |

The node resizes references to `generation_resolution × factor`, VAE-encodes
once, and attaches `{latents, strength, meta}` to the positive conditioning
as an `ic_reference_pack` key (inert to the model, same transport as
`ref_audio_bank`). Raw images in, not latents: the node owns the encode so
the downscale factor is always honored.

The module also exports `apply_ic_references(positive, negative, video_init,
pack)` — the per-chunk append routine (reference-convention
`append_keyframe` + attention entries, mirroring the official guide node's
source). The sampler imports it; the IC knowledge never lives in the sampler
file.

### 2.2 Sampler hook (the entire sampler-side footprint)

1. `_prepare_guider`: carry `ic_reference_pack` onto per-chunk conditionings
   (one line, same pattern as the ref_audio carryover).
2. Chunk builders: after conditioning assembly, if the pack is present, call
   `apply_ic_references(...)`; strip the pack key before `set_conds`.

**Correction (2026-07-27): an earlier draft claimed this node was
"independently testable" by wiring it into a plain `SamplerCustomAdvanced`.
That is wrong and contradicts §2.1/§2.2.** The pack is *inert by design* —
only the looping sampler calls `apply_ic_references()` to turn it into guide
tokens. A stock sampler never makes that call, so the pack rides along ignored
and the run shows zero reference binding. That is the architecture working, not
a failure, so the "test" can neither pass nor fail. See §4 for the real ladder:
the convention question is settled by a **tensor diff**, not by a render.

### 2.3 Per-chunk mechanics

**At build time, mirror the current `LTXAddVideoICLoRAGuide` source exactly**
— placement indices, mask values, attention entries, coordinate handling.
The convention lives in that node, not in this spec. Expected shape based on
present understanding:

1. Encode refs at downscaled resolution → `T_ref` latent frames.
2. In every chunk (first and extends), after overlap/guide/keyframe handling:
   append the reference latents as out-of-band guide tokens with the
   reference-convention indices, strength `ic_reference_strength`, plus the
   matching `_append_guide_attention_entry` records.
3. Post-sampling: references are trimmed with the other guide tokens — the
   existing `_crop_and_split` / `LTXVCropGuides` path already removes appended
   guides. **Verified 2026-07-27**: T returns exactly to its pre-injection value
   for {1,3,4} views × factor {1,2,4}. It works *because* dilation restores the
   full grid before append, so the appended tokens are a whole number of
   full-grid frames and `trim = kf_tokens // tokens_per_frame` divides evenly.
   This is the second reason `pre_filter_count` must be the larger (dilated)
   count — the crop arithmetic depends on it, not just core's mask partition.

### 2.4 Why this composes cleanly

- **Multiprompt-safe**: unlike `ref_audio` (which rode the incoming
  conditioning and was lost on per-chunk prompt replacement), the references
  are injected by the sampler *after* per-chunk conditioning is assembled, so
  a prompt swap cannot drop them.
  - **Correction (as built): "no carryover patch needed" was wrong.**
    `apply_ic_references` reads the pack off the *chunk's* positive, which
    `_prepare_guider` rebuilt from a bare encode — so without a carryover
    `get_ic_pack()` returns None on every chunk and the references vanish
    silently. The carryover is one line, mirroring `ref_audio` and
    `frame_rate`. (Reading the pack from the *base* conditioning instead would
    have made the original claim true; carrying it is the smaller change and
    keeps every per-chunk value on one path.)
- **Composes with ID-LoRA/ref_audio**: video references and audio references
  are separate token streams; both conventions can be active in one run.
- **Spatial tiling: v1 restriction.** Downscaled full-frame references do not
  correspond to spatial tile crops. v1: when `horizontal_tiles × vertical_tiles
  > 1`, warn and skip reference injection (or hard-error). Tiled+referenced is
  a later problem, if ever.

### 2.5 Cost

Per chunk: `T_ref × (H/factor) × (W/factor)` extra video tokens (the draft wrote
`·factor`; the arithmetic under it was right). Four refs at factor 2 on a
768×512 generation = 4 × 12 × 8 = **384 tokens** — small next to a chunk's own
grid, but paid in every chunk's attention.

**Two counts, easily confused** (measured, not predicted): at `factor > 1` the
guide is dilated back onto the full grid before `append_keyframe`, so the
conditioning reports `T_ref × H × W` keyframe tokens — 1536 in the example
above. Only every factor-th position is real; the holes carry mask `-1` and the
model's `grid_mask` drops them. `pre_filter_count` is deliberately the **larger**
number, because core uses it to partition the mask. The attention cost is the
smaller one.

### 2.3.1 Reference layout — the 8n+1 truncation trap

Not in the original draft, and it is a live data-loss bug in the naive path.
`LTXAddVideoICLoRAGuide.encode` truncates to `((N−1)//8)*8 + 1` pixel frames, so
a batch of reference views encodes:

| views fed | latent frames kept |
|---|---|
| 1 | 1 |
| 2, 3, 4 … 8 | **1** |
| 9 | 2 |

Four views in, three silently discarded — no error, no warning, just a weakly
conditioned run that reads as "IC references don't work very well."

`layout=one_latent_per_view` lays the stack out as `[v0, v1×8, v2×8, …]` =
`8(N−1)+1` frames, exactly `8n+1`, so nothing is dropped and each view lands on
its own latent frame (frame 0 → latent 0; each following 8-frame group → 1
latent). The node then **asserts** the encoded frame count equals the view count
rather than trusting the arithmetic. `single_frame` is the escape hatch for a
deliberately pre-composited reference sheet.

**`as_sequence` — MSR and other pre-built multiplexes.** The Licon MSR node
(LTX 2.3 Multiple-Subject-Reference) emits a reference *video* that is already
latent-aligned: subjects occupy whole latent frames, background last. At
`frame_count=41` -> 6 latents, `s1:L0-1  s2:L2  s3:L3  s4:L4  bg:L5`. That
layout IS the conditioning, so it must be encoded untouched. Running it through
`one_latent_per_view` would read each of the 41 frames as a separate view and
produce 41 latents — the multiplex destroyed and ~7x the token cost. This is
what the 2026-07-13 note meant by "MSR is incompatible with guiding-latents
slicing; long-form MSR needs per-chunk injection": MSR's reference has no
timeline correspondence, so slicing it per chunk is meaningless, but injecting
it whole into every chunk is exactly right.

## 3. Interactions and constraints

- Requires the IC-LoRA loaded via `LTXICLoRALoaderModelOnly` (metadata).
  Generic loaders leave no downscale factor to read — detect and error with a
  clear message rather than guessing.
- `optional_negative_index_latents` remains the *base-model* reference path
  (predecessor semantics, full resolution); the new input is the *trained*
  reference path. Document the distinction; using both simultaneously is
  untested and initially discouraged.
- Prompt craft: reference-driven identity means prompts should under-describe
  appearance (established rule) — mention in guide.

## 4. Test plan

The only genuinely hard question is *did we place the tokens the way the LoRA
was trained to read them* — and that is a tensor question. Renders answer it
slowly, ambiguously, and only in aggregate. Order the ladder so a typo'd index
is caught in seconds rather than inferred from a blurry face twenty minutes in.

1. **Convention diff (no sampling).** Build the guide both ways on one chunk —
   `LTXAddVideoICLoRAGuide` vs `apply_ic_references()` — with the same refs,
   factor and strength. Assert equality of: appended latent region,
   `noise_mask` region, `keyframe_idxs` (all four coord slots, start *and*
   end — the small-grid RoPE end-offset lives there), and the
   `guide_attention_entries` record (`pre_filter_count`, `latent_shape`).
   Failure here localizes to a field. Seconds, no GPU sampling.
2. **Single-chunk looper run.** Tile size ≥ total frames, so one chunk and no
   chunking. Isolates "tokens built right" from "chunking broke them." Only
   needed if (1) passes and output still looks wrong.
3. **The actual point — long-form.** Four-view turnaround prompt (front → back
   → profile → close-up) across multiple chunks: identity holds at every view,
   including views that left frame and returned. This is the test the build
   exists for; 1–2 are smoke tests that keep it interpretable.
4. **Regression**: run without refs → bit-identical to current behavior; run
   with refs + multiprompt → per-chunk prompts and refs coexist (the §2.4
   claim).
5. **Baseline for calibration**: official single-shot Ingredients workflow,
   same views → the identity-binding quality bar (3) is judged against.

## 5. Effort

Comparable to the speaker-bank build: one new node module (encode + pack +
`apply_ic_references`), a two-touch sampler hook, docs. The risky part is
matching the official node's convention exactly — budget the time to read its
source and the LoRA's expectations rather than reasoning from memory (this
spec deliberately defers to the source of truth).
