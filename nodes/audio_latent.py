import torch


class LTXAudioLatentTrim:
    """
    Trims an LTX audio latent [B, C, T, F] along the temporal dimension.
    start_index and end_index are latent frame indices (not pixel frames).
    Supports negative indexing. end_index=-1 means the last frame (inclusive).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_latent": ("LATENT",),
                "start_index": ("INT", {"default": 0, "min": -9999, "max": 9999, "step": 1}),
                "end_index": ("INT", {"default": -1, "min": -9999, "max": 9999, "step": 1}),
                "strip_mask": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "trim"
    CATEGORY = "LTXAVTools/audio"

    def trim(self, audio_latent, start_index, end_index, strip_mask=False):
        s = audio_latent.copy()
        samples = s["samples"]

        if samples.ndim != 4:
            raise ValueError(f"Expected 4D audio latent [B, C, T, F], got shape {samples.shape}")

        B, C, T, F = samples.shape
        print(f"[LTXAudioLatentTrim] input shape: {samples.shape} | T={T} frames")

        start = T + start_index if start_index < 0 else start_index
        end = T + end_index + 1 if end_index < 0 else end_index + 1

        start = max(0, min(start, T - 1))
        end = max(start + 1, min(end, T))

        s["samples"] = samples[:, :, start:end, :].contiguous()

        if strip_mask:
            s.pop("noise_mask", None)
        elif "noise_mask" in s and s["noise_mask"] is not None:
            mask = s["noise_mask"]
            if mask.ndim == 4:
                s["noise_mask"] = mask[:, :, start:end, :].contiguous()

        return (s,)


class LatentStripMask:
    """
    Removes the noise_mask from a latent dict.
    Useful before feeding latents into LTXVAddLatents to prevent mask merge errors.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "strip"
    CATEGORY = "LTXAVTools/audio"

    def strip(self, latent):
        s = latent.copy()
        s.pop("noise_mask", None)
        return (s,)


NODE_CLASS_MAPPINGS = {
    "LTXAudioLatentTrim": LTXAudioLatentTrim,
    "LatentStripMask": LatentStripMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXAudioLatentTrim": "LTX Audio Latent Trim",
    "LatentStripMask": "Latent Strip Mask",
}
