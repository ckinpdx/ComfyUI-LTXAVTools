import torch
import node_helpers


class LTXVAddAudioLatentGuide:
    """
    Adds an audio latent as a reference guide for LTX2 AV generation.
    Injects the audio as ref_audio tokens into the conditioning, placed
    before t=0 of the generation window. The model attends to these tokens
    to influence audio character/identity in the generated segment.

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
        "Injects an audio latent as reference conditioning for LTX2 AV generation. "
        "The audio is placed before t=0 of the generation window to influence audio character. "
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


NODE_CLASS_MAPPINGS = {
    "LTXVAddAudioLatentGuide": LTXVAddAudioLatentGuide,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVAddAudioLatentGuide": "LTX Add Audio Latent Guide",
}
