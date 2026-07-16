import torch
import node_helpers


class LTXVAddAudioLatentGuide:
    """
    ARTIFACT (2026-07-15): retested by the user — NO EFFECT without the
    ID-LoRA. Kept as a record, not removed. Do not build on it.

    Why it is dead: it sets ref_audio tokens on the conditioning, which the
    AV model prepends at NEGATIVE temporal coords with timestep 0
    (av_model.py _process_input) — but that placement is an ID-LoRA
    training convention. The base model was never trained to read
    negative-position audio tokens and is deaf to them. The pathway only
    functions WITH the TalkVid ID-LoRA (which is inherently single-voice;
    use the core LTXVReferenceAudio node for that).

    The working base-model alternative is CARRY-SWAP — injecting the ref
    voice into the extend chunk's audio carry slot (on-timeline, mask-
    frozen, the proven voice-transfer pathway). See SPEC_NEG_REF_AUDIO.md.

    Input must be a raw audio latent [B, C, T, F], not a NestedTensor.
    Use LTXVSeparateAVLatent first if you have a combined AV latent.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "audio_guide_latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "generate"
    CATEGORY = "LTXAVTools/audio"
    DESCRIPTION = (
        "Injects an audio latent as ref_audio conditioning tokens. The AV model "
        "prepends them in-band at negative temporal positions (before t=0, timestep 0, "
        "trimmed from output) — a negative-index audio reference, no model patch. "
        "Input must be a raw audio latent [B, C, T, F] — use LTXVSeparateAVLatent first "
        "if you have a combined AV latent."
    )

    def generate(self, positive, negative, audio_guide_latent):
        samples = audio_guide_latent["samples"]

        if samples.ndim != 4:
            raise ValueError(
                f"[LTXVAddAudioLatentGuide] Expected 4D audio latent [B, C, T, F], got shape {samples.shape}"
            )

        b, c, t, f = samples.shape
        print(f"[LTXVAddAudioLatentGuide] audio guide shape: {samples.shape} | {t} frames")

        # Reshape to token format matching LTXVReferenceAudio convention:
        # [B, C, T, F] -> [B, T, C*F]
        ref_tokens = samples.permute(0, 2, 1, 3).reshape(b, t, c * f)
        ref_audio = {"tokens": ref_tokens}

        positive = node_helpers.conditioning_set_values(positive, {"ref_audio": ref_audio})
        negative = node_helpers.conditioning_set_values(negative, {"ref_audio": ref_audio})

        return (positive, negative)


class LTXVCropAudioGuide:
    """
    Strips ref_audio from conditioning after sampling.
    Matches LTXVAddAudioLatentGuide — call this after SamplerCustomAdvanced
    to clean up the conditioning for any subsequent passes or nodes.
    No latent trimming needed (audio guides don't append tokens to the latent).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "crop"
    CATEGORY = "LTXAVTools/audio"
    DESCRIPTION = (
        "Removes ref_audio from conditioning after sampling. "
        "Use after SamplerCustomAdvanced when LTXVAddAudioLatentGuide was used, "
        "to prevent stale audio guide from affecting subsequent passes."
    )

    def crop(self, positive, negative):
        positive = node_helpers.conditioning_set_values(positive, {"ref_audio": None})
        negative = node_helpers.conditioning_set_values(negative, {"ref_audio": None})
        return (positive, negative)


NODE_CLASS_MAPPINGS = {
    "LTXVAddAudioLatentGuide": LTXVAddAudioLatentGuide,
    "LTXVCropAudioGuide": LTXVCropAudioGuide,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVAddAudioLatentGuide": "LTX Add Audio Latent Guide",
    "LTXVCropAudioGuide": "LTX Crop Audio Guide",
}
