# SPEC: Carry-Swap Reference Audio (Voice Identity)

Status: **BUILT, and the result is NEGATIVE** (built 2026-07-15+; status
corrected 2026-07-27 — the old "design confirmed, not built" line was stale).

Shipped: `LTXAVReferenceAudioBank` (`nodes/ref_audio_bank.py`) with `schedule`,
`swap_mode` (`on_change` / `always`) and `trailing_silence_frames`; the sampler's
`optional_ref_audio_bank` input; the carry-swap itself in `_sample_extend_chunk`
(`av_looping_sampler.py` ~1098), including forcing the keep-verbatim stitch on
any swapped chunk.

**Outcome: THE VOICE DOES CARRY.** Corrected 2026-07-27 — an earlier note
("base model does not adopt voice identity from context, 3 mechanisms tested")
was superseded by the working build and should not be trusted. Carry-swap
transfers the reference voice.

**Live issue: conditioning leak — generated speech does not adhere strictly to
the prompted lines.** Source not yet isolated; candidates are (a) the reference
audio's own spoken content bleeding through the carry, or (b) text-side
adherence. Prior evidence favours (a): a single-clip generation with reference
audio active was already observed producing "the character speaking the
conditioning words." See "Leak diagnosis" below.

## Goal
Per-chunk voice identity for turn-based dialog by **replacing the extend chunk's
audio carry-over with a reference-voice latent** — "reference audio as though it
was from a prior chunk." The model reads the frozen prior backwards for context;
we substitute what it hears there. Masked on the latent, not surviving the
latent. Replaces the abandoned multi-ref ID-LoRA approach
(`nodes/speaker_ref.py` ABANDONED header).

## Mechanism (confirmed design)
In `_sample_extend_chunk`, the audio init is `[carry | new]` where
`carry = audio_acc[..., -a_overlap:]` — the prior chunk's real tail, frozen by
`audio_overlap_cond_strength`. **Carry-swap replaces the carry's contents with
the scheduled ref latent:** `audio_carry = ref_voice[..., :a_overlap]`.

Everything else is existing machinery:
- **Frozen** — the existing overlap mask row already freezes the carry slot.
- **Position-correct & AV-aligned** — the carry slot is on-timeline at the chunk
  head; no coords, no length change, no model patch, no bookkeeping.
- **Not surviving** — the keep-verbatim stitch (2026-07-15) already discards the
  chunk's carry region from the output; the accumulator's real tail is what
  survives. The ref is context only.
- **In-distribution** — a frozen audio prefix is the *proven* voice-transfer
  pathway (extend chunks, KJNodes mask-extend, voices carrying over windows).
  No trained consumer required.

The change is literally *which tensor fills the carry slot*, per the schedule.

## Rejected routes (history — do not revisit without new evidence)
- **`ref_audio` conditioning tokens** (`LTXVAddAudioLatentGuide`, core
  `LTXVReferenceAudio` without LoRA): the model prepends these at NEGATIVE
  temporal coords with timestep 0 (`av_model.py:685-709`) — but that placement
  is an ID-LoRA training convention; the base model is deaf to it.
  **Empirically confirmed dead: the user retested `LTXVAddAudioLatentGuide` —
  no effect.** The pathway works only WITH the ID-LoRA (which is inherently
  single-voice). The node is kept as an artifact.
- **Literal latent prepend** (audio-only, R extra frames): audio coords are
  sequential per-latent, so the whole chunk's audio shifts late by R/25 s —
  lipsync misaligned chunk-wide. Fixing coords = re-implementing the ref_audio
  pathway. Rejected.

## Critical constraint: swap only at scripted gaps
The new region is generated to **follow the carry's tail**. With keep-verbatim,
the output splices the new region onto the accumulator's REAL tail. If the carry
was swapped mid-speech, new audio flows from the ref's ending while the output
places it after A's actual mid-word tail → **seam discontinuity in the output**.
At a choreographed turn gap (quiet beat at the seam) there is nothing to flow
from — the splice is silence-to-onset, clean.

Consequences:
- **Swap at voice changes only** (turn seams), which the operating regime makes
  every seam anyway. Same-voice seams keep the real carry.
- "Swap every seam for anti-drift" on continuous speech is RULED OUT.
- Turn-seam choreography (below) is therefore load-bearing, not optional.

## PRIMARY empirical risk (user-flagged): backwards association
The chunk still sees the frozen **video** carry — speaker A's face — while
hearing the injected ref (speaker B's voice) in the audio carry. The model may
**associate the ref voice with the prior speaker's face**: B's onset could pull
A's face into articulation, or B's voice could color toward the face it was
paired with. Wholly possible. **Must test.**

Mitigation: choreography makes the pairing coherent — a quiet, non-articulating
A face over B's voice reads as *off-screen voice, then B speaks on camera*. A
mid-articulation A face over B's voice reads as *A speaking with B's voice* —
the failure case. The gap discipline serves double duty.

## Operating regime (drives the design)
- 20 s model window; turns ≤ 20 s → **every turn fits one chunk**
  (`scene_lengths` sizes chunks to turns).
- Schedules are **pure alternation** (`1|2|1|2`) — every seam is a voice change,
  and alternation **severs the carry chain per speaker** (chunk 3's carry is
  speaker 2's audio). The ref is the ONLY identity carrier for alternating
  speakers.
- **Gender division of labor** (user-confirmed): LTX applies male/female voices
  strictly — text powers the switch categorically for mixed pairs; the ref's
  job is holding the SAME voice across a speaker's non-adjacent turns.

## Turn-seam choreography (REQUIRED — load-bearing)
1. **End A's line ~2–3 s before the chunk ends**; close the turn via the gaze
   doctrine (never closed-mouth language). The carry then shows a
   non-articulating A — no speech-in-progress prime, and the backwards
   association reads as off-screen voice.
2. **Prompt B's chunk with A as listener** — gaze/stillness only; B gets the
   articulation clause and the `[SPEECH]` line.
3. **Shot design** — favor B in frame during B's turn.

## Architecture: one sampler input, one external node (confirmed)

### `LTXAVReferenceAudioBank` (new standalone node)
- `audio_vae` (VAE); `reference_audio_1` (AUDIO, required);
  `reference_audio_2..4` (AUDIO, optional)
- `schedule` (STRING) — `1|2|1|2`, one entry per chunk, aligned with prompts /
  `scene_lengths`
- Output: `LTXAV_REF_BANK` — `{ latents: {1:…}, schedule: [...] }`
- Encodes refs to raw audio latents `[B,C,T,F]` (NOT ref_audio tokens). Each ref
  should be ≥ the audio overlap span (~2 s comfortable at overlap 48); the
  sampler slices `[..., :a_overlap]` at injection. Schedule shorter than chunk
  count → last entry repeats; invalid entry → voice 1.

### Sampler: `optional_ref_audio_bank` (appended, widget-safe)
Per extend chunk: if bank present and `schedule[i] != schedule[i-1]` (or an
explicit swap-mode says so), fill the carry slot from the bank instead of the
accumulator tail. Chunk 0: no carry slot — no injection; the prompt establishes
chunk 0's voice (schedule entry 0 exists for bookkeeping only).

## Zero-code validation available NOW (before building)
`optional_prior_av_latent` already implements "reference audio as prior chunk":
encode a short AV clip whose audio is the ref voice (video = the character, e.g.
a previous gen), feed as prior, free-generate the continuation, trim the prior
span off the output in post. Continuation experience has already half-proven
voice carry through exactly this path. A/B: continuation voice vs a no-prior
run.

## Test plan (after build)
1. **Backwards-association probe** (the user-flagged risk): mixed-gender pair,
   choreographed gap, swap at one seam. Does B onset cleanly in B's ref voice,
   or does A's face articulate / B's voice color toward A?
2. **Mixed-gender alternation** (`1|2|1|2`) — switch is text-powered, so any
   cross-turn inconsistency in one speaker's voice is a pure read on whether the
   ref anchors identity.
3. **Same-gender pair** — stress test: no categorical wall; ref must hold
   identity and out-pull the video-face prior at onsets.
4. **Sloppy-choreography control** — deliberately leave A mid-articulation at
   one swapped seam to measure how load-bearing the gap discipline is.

## Open questions (empirical)
1. Backwards association strength (risk #1 above).
2. Does a swapped carry anchor identity as strongly as a real carry does?
   (The proven mechanism transfers a voice that was genuinely there; a ref the
   video never "heard" may bind more weakly.)
3. Same-gender onset bleed.
4. Whether same-voice chunks should also swap at scripted pauses (stronger
   anchor) or always keep the real carry (better continuity) — only decidable
   by test.


## Leak diagnosis (open, 2026-07-27)

Symptom: the voice identity transfers, but the *words* drift from the prompted
lines.

**`trailing_silence_frames` defaults to 0** — check this first. The knob exists
precisely for this failure: it places N latent frames of VAE-encoded silence at
the END of the swapped carry so the new region onsets from silence rather than
continuing a mid-word reference tail. 1-3 frames (~40-120 ms) is a clean seam;
more silence weakens identity transfer. If it is at 0, the model is being handed
an unfinished utterance and asked to continue it.

**Decisive A/B for text-vs-audio:** run the same prompts with the bank
disconnected (no swap). If the words still drift, the leak is text-side —
prompt structure or dialog budget, not the reference. If they adhere cleanly
without the swap, the reference audio is the source and the fix is on this page.

**If the leak survives trailing silence,** it is the reference's content in the
BODY of the carry, not just its tail — the whole swapped span is real speech the
model reads as "what was just said." Mitigations, in order of cost: pick a
reference clip whose spoken content is short and phonetically neutral; shorten
the swapped span; or accept identity-vs-adherence as a trade and tune
`trailing_silence_frames` to taste.