import math

import torch
import comfy.model_management
from comfy.nested_tensor import NestedTensor


_LTX_VIDEO_LATENT_CHANNELS = 128
_LTX_MIN_VIDEO_SPATIAL = 4  # latent space (= 32px), safe minimum for patchifier


class LTXAudioOnlyLatent:
    """
    Creates a combined AV latent for audio-only generation.
    Pairs a minimal dummy video latent with a zero audio latent of the desired duration,
    wrapped as a NestedTensor ready for the LTX2 AV sampler.

    Wire the output latent into LTXVAudioVideoMask using audio_latent_frames to
    set up the denoising mask before sampling.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_vae": ("VAE",),
                "seconds": ("FLOAT", {
                    "default": 3.0, "min": 0.1, "max": 300.0, "step": 0.1,
                    "tooltip": "Audio duration in seconds.",
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 16, "step": 1,
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latent", "audio_latent_frames")
    FUNCTION = "create"
    CATEGORY = "LTXAVTools/audio"

    def create(self, audio_vae, seconds, batch_size):
        device = comfy.model_management.intermediate_device()

        z_channels = audio_vae.latent_channels
        freq_bins = audio_vae.latent_frequency_bins
        latents_per_second = audio_vae.latents_per_second

        num_audio_latents = math.ceil(seconds * latents_per_second)

        video = torch.zeros(
            (batch_size, _LTX_VIDEO_LATENT_CHANNELS, 1, _LTX_MIN_VIDEO_SPATIAL, _LTX_MIN_VIDEO_SPATIAL),
            device=device,
        )
        audio = torch.zeros(
            (batch_size, z_channels, num_audio_latents, freq_bins),
            device=device,
        )

        print(f"[LTXAudioOnlyLatent] audio: {audio.shape} | video dummy: {video.shape}")

        latent = {
            "samples": NestedTensor([video, audio]),
            "type": "ltxv",
        }

        return (latent, num_audio_latents)


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
    "LTXAudioOnlyLatent": LTXAudioOnlyLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXAudioLatentTrim": "LTX Audio Latent Trim",
    "LatentStripMask": "Latent Strip Mask",
    "LTXAudioOnlyLatent": "LTX Audio Only Latent",
}
