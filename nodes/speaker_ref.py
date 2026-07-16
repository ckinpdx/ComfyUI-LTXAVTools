"""
==============================================================================
ABANDONED (2026-07-15): multi-reference / multi-speaker ID-LoRA is NOT thought
to be possible with this architecture. The code below is kept as a record and
a starting point, not removed — but do not expect it to produce clean
multi-voice output.

WHY IT CANNOT WORK — the ID-LoRA is inherently SINGLE-VOICE:
  * Its ref_audio tokens are attended GLOBALLY by every audio frame (voice
    identity emerges from whole-clip self-attention), and the identity-guidance
    pass amplifies that reference GLOBALLY. There is no mechanism to say "this
    voice for these frames, that voice for those."
  * It was TRAINED to transfer ONE voice to the ENTIRE clip. Any voice
    diversity in a scene is out of distribution for it.

Per-chunk reference switching (this module's whole approach) therefore cannot
fix it: at every turn boundary the incoming speaker's global reference fights
the frozen audio carry belonging to the previous speaker, because the
conditioning is not temporally localizable. Empirically, multi-speaker runs
degrade (garbled onsets, wrong-speaker bleed), and the per-chunk-ref plus
turn-silence workarounds never fully held — they patch a mechanism that
fundamentally can't localize a single global voice.

Real multi-voice requires a DIFFERENT mechanism, not a knob on this one:
trained per-speaker binding (MultiTalk-style L-RoPE labels, or a
Bind-Your-Avatar-style embedding router), or a model-level audio-attention mask
restricting which frames attend to which ref tokens. None of that is reachable
by driving the single-speaker ID-LoRA from the sampler. See
memory/ltxav-looping-sampler.md for the full reasoning.
==============================================================================

Multi-speaker voice identity for LTX2.3 AV ID-LoRA generation.

LTXAVReferenceAudioMulti — drop-in replacement for the core LTXVReferenceAudio
node that encodes up to four reference voices. Speaker 1 behaves exactly like
the core node (default ref_audio + identity-guidance model patch); the full
set rides the positive conditioning as a `ref_audio_bank`.

LTXAVSpeakerPromptProvider — pipe-separated per-chunk prompt encoder that
parses `[SPEAKER n]:` tags, rewrites them to the ID-LoRA-trained `[SPEECH]:`
before encoding (the model never sees the routing tag), and stamps each
chunk's conditioning with `speaker_idx`.

LTXVAVLoopingSampler._prepare_guider resolves speaker_idx against the bank
per chunk. One speaker per chunk; turn-based dialog only.
"""

import re

import torchaudio

import comfy.samplers
import node_helpers


def _encode_ref_audio(audio_vae, reference_audio):
    """AUDIO dict -> ref_audio token dict. Matches core LTXVReferenceAudio."""
    sample_rate = reference_audio["sample_rate"]
    vae_sample_rate = getattr(audio_vae, "audio_sample_rate", 44100)
    waveform = reference_audio["waveform"]
    if vae_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, vae_sample_rate)
    audio_latents = audio_vae.encode(waveform.movedim(1, -1))
    b, c, t, f = audio_latents.shape
    ref_tokens = audio_latents.permute(0, 2, 1, 3).reshape(b, t, c * f)
    return {"tokens": ref_tokens}


class LTXAVReferenceAudioMulti:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model":    ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "audio_vae": ("VAE",),
                "reference_audio_1": ("AUDIO", {
                    "tooltip": "Speaker 1 — the default voice. ~5 seconds, clean, "
                               "single speaker (source-separated if possible).",
                }),
                "identity_guidance_scale": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01,
                    "tooltip": "Strength of identity guidance. Runs an extra forward "
                               "pass without reference each step. 0 disables the "
                               "extra pass entirely.",
                }),
                "start_percent": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Start of the sigma range where identity guidance is active.",
                }),
                "end_percent": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "End of the sigma range where identity guidance is active.",
                }),
            },
            "optional": {
                "reference_audio_2": ("AUDIO",),
                "reference_audio_3": ("AUDIO",),
                "reference_audio_4": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("model", "positive", "negative")
    FUNCTION     = "apply"
    CATEGORY     = "LTXAVTools/audio"
    DESCRIPTION  = (
        "Multi-speaker LTXV Reference Audio (ID-LoRA). Speaker 1 is the default "
        "voice; speakers 2-4 are selected per chunk by [SPEAKER n] tags via "
        "LTXAVSpeakerPromptProvider + LTXVAVLoopingSampler."
    )

    def apply(self, model, positive, negative, audio_vae, reference_audio_1,
              identity_guidance_scale, start_percent, end_percent,
              reference_audio_2=None, reference_audio_3=None, reference_audio_4=None):

        bank = {}
        for n, ref in enumerate(
            [reference_audio_1, reference_audio_2, reference_audio_3, reference_audio_4],
            start=1,
        ):
            if ref is not None:
                bank[n] = _encode_ref_audio(audio_vae, ref)

        print(f"[LTXAVReferenceAudioMulti] encoded speakers: {sorted(bank)}")

        default_ref = bank[1]
        positive = node_helpers.conditioning_set_values(
            positive, {"ref_audio": default_ref, "ref_audio_bank": bank}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"ref_audio": default_ref}
        )

        # Identity guidance model patch — identical to core LTXVReferenceAudio.
        # Reads whatever ref_audio is on the conditioning at sampling time, so
        # per-chunk speaker switching works with guidance automatically.
        m = model.clone()
        scale = identity_guidance_scale
        model_sampling = m.get_model_object("model_sampling")
        sigma_start = model_sampling.percent_to_sigma(start_percent)
        sigma_end = model_sampling.percent_to_sigma(end_percent)

        def post_cfg_function(args):
            if scale == 0:
                return args["denoised"]

            sigma = args["sigma"]
            sigma_ = sigma[0].item()
            if sigma_ > sigma_start or sigma_ < sigma_end:
                return args["denoised"]

            cond_pred = args["cond_denoised"]
            cond = args["cond"]
            cfg_result = args["denoised"]
            model_options = args["model_options"].copy()
            x = args["input"]

            # Strip ref_audio from conditioning for the no-reference pass
            noref_cond = []
            for entry in cond:
                new_entry = entry.copy()
                mc = new_entry.get("model_conds", {}).copy()
                mc.pop("ref_audio", None)
                new_entry["model_conds"] = mc
                noref_cond.append(new_entry)

            (pred_noref,) = comfy.samplers.calc_cond_batch(
                args["model"], [noref_cond], x, sigma, model_options
            )

            return cfg_result + (cond_pred - pred_noref) * scale

        m.set_model_sampler_post_cfg_function(post_cfg_function)

        return (m, positive, negative)


_SPEAKER_TAG = re.compile(r"\[SPEAKER\s*(\d+)\]\s*:", re.IGNORECASE)


class LTXAVSpeakerPromptProvider:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompts": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": "Prompts separated by '|', one per temporal chunk. "
                               "Use [SPEAKER n]: in place of [SPEECH]: to select "
                               "voice n from LTXAVReferenceAudioMulti for that "
                               "chunk. The tag is rewritten to [SPEECH]: before "
                               "encoding — the model never sees it. Untagged "
                               "segments use the default voice (speaker 1).",
                }),
                "clip": ("CLIP", {"tooltip": "CLIP model to encode the prompts."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditionings",)
    FUNCTION     = "get_prompt_list"
    CATEGORY     = "LTXAVTools/audio"
    DESCRIPTION  = (
        "MultiPromptProvider with [SPEAKER n] voice routing for multi-speaker "
        "ID-LoRA generation. One speaker per chunk."
    )

    def get_prompt_list(self, prompts, clip):
        conditionings = []
        for i, segment in enumerate(p.strip() for p in prompts.split("|")):
            if not segment:
                print(f"[LTXAVSpeakerPromptProvider] segment {i}: empty — skipped "
                      "(check for a trailing '|').")
                continue

            match = _SPEAKER_TAG.search(segment)
            speaker = int(match.group(1)) if match else None
            text = _SPEAKER_TAG.sub("[SPEECH]:", segment)

            cond = clip.encode_from_tokens_scheduled(clip.tokenize(text))
            if speaker is not None:
                cond = node_helpers.conditioning_set_values(cond, {"speaker_idx": speaker})
            conditionings.append(cond)

            # Log the encoded speech span so tag rewriting is verifiable per run.
            label = f"SPEAKER {speaker}" if speaker is not None else "default"
            sp = text.find("[SPEECH]:")
            speech_head = text[sp:sp + 80].replace("\n", " ") if sp != -1 else "<no [SPEECH] section>"
            print(f"[LTXAVSpeakerPromptProvider] segment {i}: {label} | {speech_head}")

        return (conditionings,)


NODE_CLASS_MAPPINGS = {
    "LTXAVReferenceAudioMulti":   LTXAVReferenceAudioMulti,
    "LTXAVSpeakerPromptProvider": LTXAVSpeakerPromptProvider,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXAVReferenceAudioMulti":   "LTX AV Reference Audio Multi (ID-LoRA)",
    "LTXAVSpeakerPromptProvider": "LTX AV Speaker Prompt Provider",
}
