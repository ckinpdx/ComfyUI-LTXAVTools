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


class LTXAVExtendLatent:
    """
    Prepares an AV NestedTensor latent for video extension.

    Takes an encoded AV latent (existing content) and appends zero latents
    for the extension region, with a noise_mask pre-built:
      - existing_denoise  for existing frames (0.0 = fully preserve)
      - 1.0              for new frames (fully generate)

    Wire extension_start_frame into optional_cond_image_indices on the
    looping sampler to place a keyframe at the transition boundary.
    The transition image should be the last frame of the existing video,
    so the model has a hard visual anchor to continue from.
    """

    _AUDIO_LATENTS_PER_SECOND = 25.0

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "av_latent": ("LATENT",),
                "vae": ("VAE",),
                "extension_seconds": ("FLOAT", {
                    "default": 3.0, "min": 0.1, "max": 300.0, "step": 0.1,
                    "tooltip": "Duration to add beyond the end of the existing video.",
                }),
                "fps": ("FLOAT", {
                    "default": 25.0, "min": 1.0, "max": 120.0, "step": 0.01,
                    "tooltip": "Must match the fps of the AV latent.",
                }),
                "existing_denoise": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "Noise mask value for existing frames. "
                        "0.0 = fully preserve. "
                        "Small values (0.1–0.3) allow light refinement near the transition."
                    ),
                }),
            }
        }

    RETURN_TYPES  = ("LATENT", "INT", "INT")
    RETURN_NAMES  = ("av_latent", "extension_start_frame", "last_existing_frame")
    FUNCTION      = "extend"
    CATEGORY      = "LTXAVTools/utils"
    DESCRIPTION   = (
        "Appends zero latents to an existing AV latent for video extension. "
        "Sets noise_mask so the sampler preserves existing content and freely "
        "generates the new region. extension_start_frame and last_existing_frame "
        "are pixel-frame indices for use with optional_cond_image_indices."
    )

    def extend(self, av_latent, vae, extension_seconds, fps, existing_denoise):
        samples = av_latent["samples"]
        if not isinstance(samples, NestedTensor):
            raise ValueError(
                "[LTXAVExtendLatent] Input must be an AV NestedTensor latent. "
                "Use LTXVSeparateAVLatent or encode with the AV VAE first."
            )

        time_sc = vae.downscale_index_formula[0]  # 8 for LTX

        video = samples.tensors[0]   # [B, C_v, T_v, H, W]
        audio = samples.tensors[1]   # [B, C_a, T_a, F_s]
        B, C_v, T_v, H, W = video.shape
        _,  C_a, T_a, F_s = audio.shape
        dev, dty = video.device, video.dtype

        # Extension video latent frames — each non-first latent = time_sc pixel frames
        ext_px            = max(time_sc, round(round(extension_seconds * fps) / time_sc) * time_sc)
        ext_video_latents = ext_px // time_sc

        # Combined totals
        combined_T_v  = T_v + ext_video_latents
        total_px      = (combined_T_v - 1) * time_sc + 1
        total_audio   = round(total_px / fps * self._AUDIO_LATENTS_PER_SECOND)
        ext_audio     = max(0, total_audio - T_a)

        # Build combined tensors
        video_combined = torch.cat(
            [video, torch.zeros(B, C_v, ext_video_latents, H, W, device=dev, dtype=dty)],
            dim=2,
        )
        audio_combined = torch.cat(
            [audio, torch.zeros(B, C_a, ext_audio, F_s, device=dev, dtype=dty)],
            dim=2,
        )

        # Noise mask: existing_denoise for existing, 1.0 for new
        video_mask = torch.ones(B, 1, combined_T_v, 1, 1, device=dev, dtype=dty)
        video_mask[:, :, :T_v] = existing_denoise
        audio_mask = torch.ones(B, 1, total_audio, F_s, device=dev, dtype=dty)
        audio_mask[:, :, :T_a] = existing_denoise

        out = {
            **{k: v for k, v in av_latent.items() if k not in ("samples", "noise_mask")},
            "samples":    NestedTensor([video_combined, audio_combined]),
            "noise_mask": NestedTensor([video_mask, audio_mask]),
        }

        # Pixel-frame indices for keyframe conditioning
        # last_existing_frame  = last pixel frame of existing content
        # extension_start_frame = first pixel frame of new content
        last_existing_frame   = (T_v - 1) * time_sc
        extension_start_frame = last_existing_frame + 1

        print(
            f"[LTXAVExtendLatent] existing {T_v}v/{T_a}a latents "
            f"+ extension {ext_video_latents}v/{ext_audio}a latents "
            f"| last_existing_frame={last_existing_frame} "
            f"| extension_start_frame={extension_start_frame}"
        )

        return (out, extension_start_frame, last_existing_frame)


NODE_CLASS_MAPPINGS = {
    "LTXAudioLatentTrim":  LTXAudioLatentTrim,
    "LatentStripMask":     LatentStripMask,
    "LTXAudioOnlyLatent":  LTXAudioOnlyLatent,
    "LTXAVExtendLatent":   LTXAVExtendLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXAudioLatentTrim":  "LTX Audio Latent Trim",
    "LatentStripMask":     "Latent Strip Mask",
    "LTXAudioOnlyLatent":  "LTX Audio Only Latent",
    "LTXAVExtendLatent":   "LTX AV Extend Latent",
}
