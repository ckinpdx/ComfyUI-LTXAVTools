import os
import uuid
import numpy as np
import torch
import folder_paths
from comfy import model_management
from PIL import Image

try:
    from comfy.nested_tensor import NestedTensor
    _HAS_NESTED = True
except ImportError:
    _HAS_NESTED = False


class PreviewImagePassthrough:
    """
    Displays a preview of the input image and passes it through unchanged.
    Useful inside loops where terminal PreviewImage nodes don't refresh per iteration.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "preview"
    OUTPUT_NODE   = True
    CATEGORY      = "LTXAVTools/utils"

    def preview(self, image):
        tmp_dir = folder_paths.get_temp_directory()
        results = []

        for i in range(image.shape[0]):
            arr = (image[i].numpy() * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
            filename = f"preview_{uuid.uuid4().hex[:12]}.png"
            path = os.path.join(tmp_dir, filename)
            img.save(path)
            results.append({
                "filename": filename,
                "subfolder": "",
                "type": "temp",
            })

        return {"ui": {"images": results}, "result": (image,)}


class LTXAVLatentCheck:
    """
    Checks whether the video and audio components of an LTX AV nested latent
    are time-matched for a given fps. Reports actual vs expected audio latent
    frames and the delta. Passes the latent through unchanged.

    Expected relationship: audio_latent_frames = 8 * video_latent_frames - 7
    (derived from LTX temporal compression: first video latent = 1 pixel frame,
    subsequent = 8 pixel frames each; at fps == audio_latents_per_second == 25
    this equals frame_count exactly).
    """

    AUDIO_LATENTS_PER_SECOND = 25.0

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "fps": ("FLOAT", {
                    "default": 25.0, "min": 1.0, "max": 120.0, "step": 0.01,
                    "tooltip": "Video fps used to compute expected audio latent count.",
                }),
            }
        }

    RETURN_TYPES  = ("LATENT", "INT", "INT", "INT", "INT", "BOOLEAN")
    RETURN_NAMES  = ("latent", "video_latent_frames", "audio_latent_frames", "expected_audio_frames", "delta", "is_matched")
    FUNCTION      = "check"
    OUTPUT_NODE   = True
    CATEGORY      = "LTXAVTools/utils"

    def check(self, latent, fps):
        samples = latent["samples"]

        if _HAS_NESTED and isinstance(samples, NestedTensor):
            video = samples.tensors[0]  # [B, C, T_v, H, W]
            audio = samples.tensors[1]  # [B, C, T_a, F]
            T_v = int(video.shape[2])
            T_a = int(audio.shape[2])
        else:
            # Plain video latent — no audio to compare
            T_v = int(samples.shape[2])
            T_a = 0

        # Expected audio latents: pixel_frames / fps * audio_latents_per_second
        # pixel_frames = (T_v - 1) * 8 + 1
        pixel_frames = (T_v - 1) * 8 + 1
        expected = round(pixel_frames / fps * self.AUDIO_LATENTS_PER_SECOND)
        delta = T_a - expected
        matched = delta == 0

        status = "OK" if matched else f"MISMATCH delta={delta:+d}"
        print(f"[LTXAVLatentCheck] video={T_v} latents | audio={T_a} latents | expected={expected} | {status}")

        return (latent, T_v, T_a, expected, delta, matched)


class LTXAVSeparateCheck:
    """
    Checks time alignment between separate video and audio latents.
    Same math as LTXAVLatentCheck but accepts the latents split rather than nested.
    Place after trim operations to verify video and audio are still in sync.
    """

    AUDIO_LATENTS_PER_SECOND = 25.0

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_latent": ("LATENT",),
                "audio_latent": ("LATENT",),
                "fps": ("FLOAT", {
                    "default": 25.0, "min": 1.0, "max": 120.0, "step": 0.01,
                }),
            }
        }

    RETURN_TYPES  = ("LATENT", "LATENT", "INT", "INT", "INT", "INT", "BOOLEAN")
    RETURN_NAMES  = ("video_latent", "audio_latent", "video_latent_frames", "audio_latent_frames", "expected_audio_frames", "delta", "is_matched")
    FUNCTION      = "check"
    OUTPUT_NODE   = True
    CATEGORY      = "LTXAVTools/utils"

    def check(self, video_latent, audio_latent, fps):
        T_v = int(video_latent["samples"].shape[2])
        T_a = int(audio_latent["samples"].shape[2])

        pixel_frames = (T_v - 1) * 8 + 1
        expected = round(pixel_frames / fps * self.AUDIO_LATENTS_PER_SECOND)
        delta = T_a - expected
        matched = delta == 0

        status = "OK" if matched else f"MISMATCH delta={delta:+d}"
        print(f"[LTXAVSeparateCheck] video={T_v} latents | audio={T_a} latents | expected={expected} | {status}")

        return (video_latent, audio_latent, T_v, T_a, expected, delta, matched)


class LTXAudioLatentPad:
    """
    Pads an audio latent [B, C, T, F] by repeating the last frame N times.
    Use inside sliding-window loops before accumulation to close the 7-frame
    audio gap that appears at every concatenation boundary due to LTX's
    first-frame asymmetry (first latent = 1 pixel frame, all others = 8).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_latent": ("LATENT",),
                "pad_frames": ("INT", {
                    "default": 7, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Number of frames to append by repeating the last frame. Use 7 to fix concatenation boundary drift.",
                }),
            }
        }

    RETURN_TYPES  = ("LATENT",)
    RETURN_NAMES  = ("audio_latent",)
    FUNCTION      = "pad"
    CATEGORY      = "LTXAVTools/utils"

    def pad(self, audio_latent, pad_frames):
        samples = audio_latent["samples"]  # [B, C, T, F]
        last = samples[:, :, -1:, :]       # [B, C, 1, F]
        padding = last.expand(-1, -1, pad_frames, -1)
        padded = torch.cat([samples, padding], dim=2)
        out = {**audio_latent, "samples": padded}
        if "noise_mask" in out:
            del out["noise_mask"]
        return (out,)


class LTXVAVLatentUpsampler:
    """
    AV-aware wrapper around the LTX latent upscale model with CPU fallback.

    The LTX upsampler uses Conv3d + GroupNorm throughout. GroupNorm normalises
    across T×H×W jointly, so temporal chunking changes the statistics and
    causes seam artefacts regardless of overlap size. The full tensor must be
    processed at once. This node tries GPU first; if it OOMs it retries on CPU.

    Handles both plain video latents [B, C, T, H, W] and AV NestedTensors —
    only the video component is upsampled; audio passes through unchanged.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples":       ("LATENT",),
                "upscale_model": ("LATENT_UPSCALE_MODEL",),
                "vae":           ("VAE",),
            }
        }

    RETURN_TYPES  = ("LATENT",)
    FUNCTION      = "upsample_latent"
    CATEGORY      = "LTXAVTools/utils"

    def upsample_latent(self, samples, upscale_model, vae):
        raw   = samples["samples"]
        is_av = _HAS_NESTED and isinstance(raw, NestedTensor)

        if is_av:
            video = raw.tensors[0]   # [B, C, T, H, W]
            audio = raw.tensors[1]   # passed through unchanged
        else:
            video = raw
            audio = None

        stats       = vae.first_stage_model.per_channel_statistics
        model_dtype = next(upscale_model.parameters()).dtype
        input_dtype = video.dtype

        video_un = stats.un_normalize(video).to(dtype=model_dtype)
        print(f"[LTXVLatentUpsamplerTiled] input {tuple(video_un.shape)}")

        device = model_management.get_torch_device()
        upscale_model.to(device)
        try:
            upsampled = upscale_model(video_un.to(device))
        except torch.cuda.OutOfMemoryError:
            print(
                "[LTXVLatentUpsamplerTiled] GPU OOM — retrying on CPU (this will be slow)."
            )
            upscale_model.cpu()
            upsampled = upscale_model(video_un.cpu())
        finally:
            upscale_model.cpu()

        upsampled = stats.normalize(upsampled).to(
            dtype=input_dtype,
            device=model_management.intermediate_device(),
        )

        out = samples.copy()
        out.pop("noise_mask", None)

        if is_av:
            out["samples"] = NestedTensor([upsampled, audio.to(upsampled.device)])
        else:
            out["samples"] = upsampled

        return (out,)


class LTXVAVLatentUpsamplerTiled:
    """
    Temporally tiled version of the LTX AV latent upsampler.

    Splits the video latent into overlapping temporal tiles, upsamples each
    on GPU, and blends them back with a linear crossfade. Viable when the
    upsampled latent feeds a low-sigma refinement pass, which smooths over
    any residual tiling statistics differences.

    Use the non-tiled LTX AV Latent Upsampler instead when you need to
    process the full tensor in one shot (with CPU fallback).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples":       ("LATENT",),
                "upscale_model": ("LATENT_UPSCALE_MODEL",),
                "vae":           ("VAE",),
                "tile_frames": ("INT", {
                    "default": 16, "min": 2, "max": 256, "step": 1,
                    "tooltip": "Latent frames per temporal tile.",
                }),
                "tile_overlap": ("INT", {
                    "default": 4, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Latent frames of overlap between tiles used for blending.",
                }),
            }
        }

    RETURN_TYPES  = ("LATENT",)
    FUNCTION      = "upsample_latent"
    CATEGORY      = "LTXAVTools/utils"

    def upsample_latent(self, samples, upscale_model, vae, tile_frames, tile_overlap):
        raw   = samples["samples"]
        is_av = _HAS_NESTED and isinstance(raw, NestedTensor)

        if is_av:
            video = raw.tensors[0]   # [B, C, T, H, W]
            audio = raw.tensors[1]
        else:
            video = raw
            audio = None

        stats       = vae.first_stage_model.per_channel_statistics
        model_dtype = next(upscale_model.parameters()).dtype
        input_dtype = video.dtype
        inter_dev   = model_management.intermediate_device()
        gpu_dev     = model_management.get_torch_device()

        T    = video.shape[2]
        step = max(1, tile_frames - tile_overlap)

        result         = None
        result_weights = None

        upscale_model.to(gpu_dev)
        try:
            t_start = 0
            while t_start < T:
                t_end  = min(t_start + tile_frames, T)
                tile_v = video[:, :, t_start:t_end]

                tile_un  = stats.un_normalize(tile_v).to(dtype=model_dtype, device=gpu_dev)
                up_tile  = upscale_model(tile_un)
                up_tile  = stats.normalize(up_tile).to(dtype=input_dtype, device=inter_dev)

                if result is None:
                    B, C, _, H_up, W_up = up_tile.shape
                    result         = torch.zeros(B, C, T, H_up, W_up,
                                                 device=inter_dev, dtype=input_dtype)
                    result_weights = torch.zeros(B, 1, T, 1, 1,
                                                 device=inter_dev, dtype=input_dtype)

                tile_T  = t_end - t_start
                w       = torch.ones(tile_T, device=inter_dev, dtype=input_dtype)
                if t_start > 0:
                    w[:tile_overlap] = torch.linspace(0, 1, tile_overlap,
                                                      device=inter_dev, dtype=input_dtype)
                if t_end < T:
                    w[-tile_overlap:] = torch.minimum(
                        w[-tile_overlap:],
                        torch.linspace(1, 0, tile_overlap, device=inter_dev, dtype=input_dtype),
                    )

                w = w.view(1, 1, tile_T, 1, 1)
                result[:, :, t_start:t_end]         += up_tile * w
                result_weights[:, :, t_start:t_end] += w

                print(f"[LTXVAVLatentUpsamplerTiled] tile [{t_start},{t_end}) "
                      f"of {T} latent frames")

                if t_end >= T:
                    break
                t_start += step
        finally:
            upscale_model.cpu()

        result = result / (result_weights + 1e-8)

        out = samples.copy()
        out.pop("noise_mask", None)

        if is_av:
            out["samples"] = NestedTensor([result, audio.to(result.device)])
        else:
            out["samples"] = result

        return (out,)


NODE_CLASS_MAPPINGS = {
    "PreviewImagePassthrough":          PreviewImagePassthrough,
    "LTXAVLatentCheck":                 LTXAVLatentCheck,
    "LTXAVSeparateCheck":               LTXAVSeparateCheck,
    "LTXAudioLatentPad":                LTXAudioLatentPad,
    "LTXVAVLatentUpsampler":            LTXVAVLatentUpsampler,
    "LTXVAVLatentUpsamplerTiled":       LTXVAVLatentUpsamplerTiled,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PreviewImagePassthrough":          "Preview Image Passthrough",
    "LTXAVLatentCheck":                 "LTX AV Latent Check",
    "LTXAVSeparateCheck":               "LTX AV Separate Check",
    "LTXAudioLatentPad":                "LTX Audio Latent Pad",
    "LTXVAVLatentUpsampler":            "LTX AV Latent Upsampler",
    "LTXVAVLatentUpsamplerTiled":       "LTX AV Latent Upsampler (Tiled)",
}
