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
                "head_trim": ("INT", {
                    "default": 2, "min": 1, "max": 8, "step": 1,
                    "tooltip": "TEMPORAL upscalers only (auto-detected; ignored in "
                               "spatial mode): output latents dropped from each "
                               "non-first tile's head. Tile heads are malformed "
                               "video-start latents (first-frame asymmetry) — the "
                               "previous tile owns that region instead. Raise if "
                               "tile joins show motion stutter (raise tile_overlap "
                               "with it: blend span = 2*overlap-1-head_trim).",
                }),
            }
        }

    RETURN_TYPES  = ("LATENT",)
    FUNCTION      = "upsample_latent"
    CATEGORY      = "LTXAVTools/utils"

    def upsample_latent(self, samples, upscale_model, vae, tile_frames, tile_overlap,
                        head_trim=2):
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
        temporal_mode  = None   # auto-detected from the first tile's output shape
        T_out          = None

        upscale_model.to(gpu_dev)
        try:
            t_start = 0
            while t_start < T:
                t_end  = min(t_start + tile_frames, T)
                tile_v = video[:, :, t_start:t_end]
                L      = t_end - t_start

                tile_un  = stats.un_normalize(tile_v).to(dtype=model_dtype, device=gpu_dev)
                up_tile  = upscale_model(tile_un)
                up_tile  = stats.normalize(up_tile).to(dtype=input_dtype, device=inter_dev)

                # Mode detection (SPEC_TILED_TEMPORAL.md): spatial upscalers keep
                # T (L -> L); the temporal upscaler doubles the pixel timeline
                # (L -> 2L-1 latents, first-frame asymmetry).
                up_T = up_tile.shape[2]
                if temporal_mode is None:
                    if up_T == L:
                        temporal_mode = False
                        T_out = T
                    elif up_T == 2 * L - 1:
                        temporal_mode = True
                        T_out = 2 * T - 1
                        if 2 * tile_overlap - 1 - head_trim < 1:
                            raise ValueError(
                                f"[LTXVAVLatentUpsamplerTiled] temporal mode needs a "
                                f"blend span of at least 1 latent: 2*tile_overlap-1-"
                                f"head_trim = {2 * tile_overlap - 1 - head_trim}. "
                                f"Raise tile_overlap or lower head_trim."
                            )
                        print(f"[LTXVAVLatentUpsamplerTiled] TEMPORAL upscaler "
                              f"detected ({L} -> {up_T}): output {T_out} latents, "
                              f"head_trim {head_trim}.")
                    else:
                        raise ValueError(
                            f"[LTXVAVLatentUpsamplerTiled] unsupported temporal "
                            f"mapping {L} -> {up_T}. Supported: L -> L (spatial) "
                            f"and L -> 2L-1 (temporal 2x)."
                        )
                else:
                    expected = (2 * L - 1) if temporal_mode else L
                    if up_T != expected:
                        raise ValueError(
                            f"[LTXVAVLatentUpsamplerTiled] inconsistent tile output: "
                            f"expected {expected} latents for a {L}-latent tile, got {up_T}."
                        )

                if result is None:
                    B, C, _, H_up, W_up = up_tile.shape
                    result         = torch.zeros(B, C, T_out, H_up, W_up,
                                                 device=inter_dev, dtype=input_dtype)
                    result_weights = torch.zeros(B, 1, T_out, 1, 1,
                                                 device=inter_dev, dtype=input_dtype)

                if not temporal_mode:
                    # --- spatial path (unchanged) ---
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
                else:
                    # --- temporal path ---
                    # Anchor mapping: input latent t <-> output latent 2t. A tile
                    # at t_start lands at output [2*t_start, 2*t_start + 2L-1).
                    # Non-first tiles: drop head_trim malformed head latents (tile
                    # heads are video-start latents); the previous tile owns that
                    # region. Ramp-in spans the remaining output overlap and ends
                    # exactly where the previous tile's data ends.
                    g0   = 2 * t_start
                    trim = head_trim if t_start > 0 else 0
                    tile_out = up_tile[:, :, trim:]
                    g0  += trim
                    written = tile_out.shape[2]
                    ov_out  = 2 * tile_overlap - 1

                    w = torch.ones(written, device=inter_dev, dtype=input_dtype)
                    if t_start > 0:
                        blend_in = ov_out - trim
                        w[:blend_in] = torch.linspace(0, 1, blend_in,
                                                      device=inter_dev, dtype=input_dtype)
                    if t_end < T:
                        w[-ov_out:] = torch.minimum(
                            w[-ov_out:],
                            torch.linspace(1, 0, ov_out, device=inter_dev, dtype=input_dtype),
                        )

                    w = w.view(1, 1, written, 1, 1)
                    result[:, :, g0:g0 + written]         += tile_out * w
                    result_weights[:, :, g0:g0 + written] += w

                    print(f"[LTXVAVLatentUpsamplerTiled] temporal tile "
                          f"[{t_start},{t_end}) of {T} -> out [{g0},{g0 + written}) "
                          f"of {T_out} (trim {trim})")

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


class LTXKeyframePairConcat:
    """
    Emits consecutive keyframe pairs as one image, for vision-LLM prompting.

    Cycle 1 concatenates keyframes 1+2, cycle 2 -> 2+3, and so on — pair k is
    exactly scene k's travel endpoints under the end-anchored keyframe plan
    (LTXKeyframePlanner), so a VLM shown the pair can write scene k's
    transition prompt. Drive `index` with an incrementing INT primitive
    (control_after_generate) to walk the batch across queue cycles;
    `total_pairs` gives the cycle bound.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Keyframe batch, in plan order."}),
                "index": ("INT", {
                    "default": 1, "min": 1, "max": 10000,
                    "tooltip": "1-based pair index: 1 -> keyframes 1+2, 2 -> 2+3… "
                               "Clamped to the last valid pair.",
                }),
                "direction": (["horizontal", "vertical"], {
                    "default": "horizontal",
                    "tooltip": "horizontal: earlier keyframe LEFT, later RIGHT. "
                               "vertical: earlier TOP, later BOTTOM.",
                }),
                "gap": ("INT", {
                    "default": 8, "min": 0, "max": 128, "step": 1,
                    "tooltip": "Black divider between the two panels (pixels). "
                               "Helps a VLM read them as two distinct panels.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("image", "pair_info", "total_pairs")
    FUNCTION     = "concat"
    CATEGORY     = "LTXAVTools/utils"
    DESCRIPTION  = (
        "Concatenates consecutive images from a batch (index 1 -> images 1+2, "
        "index 2 -> 2+3…) for vision-LLM scene/transition prompting. Pair k = "
        "scene k's endpoints under the end-anchored keyframe plan."
    )

    def concat(self, images, index, direction, gap):
        n = images.shape[0]
        if n < 2:
            print("[LTXKeyframePairConcat] batch has fewer than 2 images — "
                  "passing the single image through.")
            return (images[:1], "single image (no pair)", 0)

        total_pairs = n - 1
        i = max(1, min(index, total_pairs))
        if i != index:
            print(f"[LTXKeyframePairConcat] index {index} clamped to {i} "
                  f"(batch of {n} -> {total_pairs} pairs).")

        a = images[i - 1]   # [H, W, C]
        b = images[i]
        dim = 1 if direction == "horizontal" else 0

        parts = [a]
        if gap > 0:
            gap_shape = list(a.shape)
            gap_shape[dim] = gap
            parts.append(torch.zeros(gap_shape, device=a.device, dtype=a.dtype))
        parts.append(b)

        out  = torch.cat(parts, dim=dim).unsqueeze(0)
        info = f"pair {i}/{total_pairs}: keyframe {i} -> {i + 1} ({direction})"
        print(f"[LTXKeyframePairConcat] {info}")
        return (out, info, total_pairs)


class LTXLoraMetadataReader:
    """
    Single-selection LoRA metadata reader for IC-LoRA workflows.

    Reads only the safetensors JSON header (no tensor loading — milliseconds,
    no VRAM) and emits the absolute path alongside the metadata, so ONE combo
    drives both the loader and the sampler:

        Metadata Reader ── lora_path ──▶ KJ LTX2 LoRA Loader Advanced
                       │                 (opt_lora_path overrides its combo)
                       └─ latent_downscale_factor ──▶ sampler
                          guiding_downscale_factor

    The factor comes from the LoRA's own reference_downscale_factor metadata
    (pixel spatial upscaler x2 = 2, x4 = 4) — no manual sync, no drift.
    """

    @classmethod
    def INPUT_TYPES(s):
        import folder_paths
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "The single point of LoRA selection: path feeds the "
                               "loader, factor feeds the sampler.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("lora_path", "latent_downscale_factor", "metadata")
    FUNCTION     = "read"
    CATEGORY     = "LTXAVTools/utils"
    DESCRIPTION  = (
        "Reads a LoRA's safetensors metadata header (no weight loading). Outputs "
        "the absolute path (wire to a loader's opt_lora_path so one combo drives "
        "everything), the IC-LoRA reference_downscale_factor (wire to the AV "
        "Looping Sampler's guiding_downscale_factor), and the full metadata for "
        "inspection."
    )

    def read(self, lora_name):
        import json
        import struct
        import folder_paths

        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len).decode("utf-8"))
        md = header.get("__metadata__", {}) or {}

        try:
            factor = max(1.0, float(md.get("reference_downscale_factor", 1)))
        except (TypeError, ValueError):
            factor = 1.0

        meta_str = json.dumps(md, indent=2) if md else "(no metadata)"
        print(f"[LTXLoraMetadataReader] {lora_name}: "
              f"reference_downscale_factor={factor} | {len(md)} metadata keys")
        return (path, factor, meta_str)


class LTXAVStreamingSave:
    """
    Chunked VAE decode streamed straight into ffmpeg — the full pixel tensor
    never exists. Constant RAM regardless of video length: only one chunk of
    frames is alive at any moment, piped rawvideo into a persistent encoder.

    Exactness: the LTX video VAE is CAUSAL (past-context only), so decoding a
    slice with `context_latents` of left context and trimming the context's
    pixels yields the same frames a full decode would — no right context, no
    crossfade. The trim also absorbs the slice's first-frame asymmetry (its
    first latent decodes as a 1-px video start, which lands in the discarded
    region). Total streamed frames = (T-1)*8+1, identical to a full decode.

    Audio is NOT decoded here (it is tiny — use LTXVAudioVAEDecode) — feed the
    decoded AUDIO in and it is muxed into the file at the end.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "AV NestedTensor (video component is used) or a "
                               "plain 5D video latent.",
                }),
                "vae": ("VAE",),
                "chunk_latents": ("INT", {
                    "default": 16, "min": 2, "max": 256,
                    "tooltip": "Latent frames decoded per chunk (~ chunk*8 pixel "
                               "frames of RAM at a time).",
                }),
                "context_latents": ("INT", {
                    "default": 4, "min": 1, "max": 16,
                    "tooltip": "Left-context latents decoded with each chunk and "
                               "trimmed. Must cover the causal VAE's temporal "
                               "receptive field; raise if you ever see a subtle "
                               "seam at chunk boundaries.",
                }),
                "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "filename_prefix": ("STRING", {"default": "LTXAV/stream"}),
                "crf": ("INT", {
                    "default": 19, "min": 0, "max": 51,
                    "tooltip": "libx264 quality (lower = better/larger).",
                }),
            },
            "optional": {
                "optional_audio": ("AUDIO", {
                    "tooltip": "Decoded audio (LTXVAudioVAEDecode) to mux into "
                               "the file. Omit for silent video.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    OUTPUT_NODE  = True
    FUNCTION     = "stream_save"
    CATEGORY     = "LTXAVTools/utils"
    DESCRIPTION  = (
        "Long-video export without the RAM cliff: decodes the video latent in "
        "chunks (causal-context-exact) and streams frames directly into ffmpeg. "
        "The full pixel tensor never exists — RAM use is constant at any length. "
        "Feed decoded AUDIO to mux; audio decode is cheap and stays external."
    )

    def stream_save(self, latent, vae, chunk_latents, context_latents, fps,
                    filename_prefix, crf, optional_audio=None):
        import os
        import shutil
        import subprocess
        import folder_paths

        raw = latent["samples"]
        if _HAS_NESTED and isinstance(raw, NestedTensor):
            video = raw.tensors[0]
        else:
            video = raw
        if video.ndim != 5:
            raise ValueError(
                f"[LTXAVStreamingSave] expected a 5D video latent, got {video.ndim}D."
            )
        video = video[:1]
        T = video.shape[2]

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            try:
                from imageio_ffmpeg import get_ffmpeg_exe
                ffmpeg = get_ffmpeg_exe()
            except Exception:
                raise RuntimeError(
                    "[LTXAVStreamingSave] ffmpeg not found on PATH (and "
                    "imageio-ffmpeg unavailable)."
                )
        # Logged so shadowed-PATH problems (conda/minimal builds without
        # libx264) are visible in reports.
        print(f"[LTXAVStreamingSave] using ffmpeg: {ffmpeg}")

        out_dir = folder_paths.get_output_directory()
        full_folder, fname, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, out_dir
        )
        video_tmp  = os.path.join(full_folder, f"{fname}_{counter:05}_tmp.mp4")
        final_path = os.path.join(full_folder, f"{fname}_{counter:05}.mp4")

        import tempfile

        proc = None
        frames_written = 0
        # ffmpeg's stderr goes to a temp file, not a pipe: nothing reads the
        # pipe during the (long) encode, so a chatty ffmpeg could fill it and
        # deadlock. The file is read back only to build error messages.
        err_file = tempfile.TemporaryFile()

        def _err_tail():
            try:
                err_file.seek(0)
                tail = err_file.read()[-2000:].decode("utf-8", "replace").strip()
            except Exception:
                tail = ""
            return (f"\n--- ffmpeg stderr ---\n{tail}" if tail
                    else " (ffmpeg printed no error output)")

        try:
            k = 0
            while k < T:
                model_management.throw_exception_if_processing_interrupted()
                n = min(chunk_latents, T - k)
                c = 0 if k == 0 else min(context_latents, k)
                px = vae.decode(video[:, :, k - c : k + n])
                if isinstance(px, tuple):
                    px = px[0]
                if px.ndim == 5:
                    px = px.reshape(-1, *px.shape[-3:])
                if k > 0:
                    # keep exactly this chunk's 8*n frames; the discarded head
                    # holds the context latents' pixels incl. the malformed
                    # 1-px slice start.
                    px = px[-(8 * n):]

                if proc is None:
                    H, W = int(px.shape[1]), int(px.shape[2])
                    proc = subprocess.Popen(
                        [ffmpeg, "-y", "-loglevel", "error",
                         "-f", "rawvideo", "-pix_fmt", "rgb24",
                         "-s", f"{W}x{H}", "-r", str(fps), "-i", "pipe:",
                         "-c:v", "libx264", "-preset", "medium",
                         "-crf", str(crf), "-pix_fmt", "yuv420p",
                         video_tmp],
                        stdin=subprocess.PIPE,
                        stderr=err_file,
                    )

                data = (
                    px.clamp(0, 1).mul(255).round()
                      .to(torch.uint8).cpu().contiguous().numpy().tobytes()
                )
                try:
                    proc.stdin.write(data)
                except (BrokenPipeError, OSError):
                    # ffmpeg died mid-stream (bad build, permissions, disk
                    # full) — the pipe error is just the messenger; surface
                    # ffmpeg's own words instead.
                    try:
                        proc.stdin.close()
                    except Exception:
                        pass
                    ret = proc.wait()
                    proc = None
                    raise RuntimeError(
                        f"[LTXAVStreamingSave] ffmpeg died mid-stream "
                        f"(exit {ret}).{_err_tail()}"
                    )
                frames_written += px.shape[0]
                print(f"[LTXAVStreamingSave] latents [{k},{k + n}) of {T} -> "
                      f"{px.shape[0]} frames (total {frames_written})")
                del px, data
                k += n

            proc.stdin.close()
            ret = proc.wait()
            if ret != 0:
                raise RuntimeError(
                    f"[LTXAVStreamingSave] ffmpeg exited with {ret}.{_err_tail()}"
                )
            proc = None
        finally:
            if proc is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                proc.kill()
            err_file.close()

        if optional_audio is not None and optional_audio.get("waveform") is not None:
            import torchaudio
            wav_tmp = os.path.join(full_folder, f"{fname}_{counter:05}_tmp.wav")
            wf = optional_audio["waveform"]
            if wf.ndim == 3:
                wf = wf[0]
            torchaudio.save(wav_tmp, wf.cpu(), int(optional_audio["sample_rate"]))
            try:
                mux = subprocess.run(
                    [ffmpeg, "-y", "-loglevel", "error",
                     "-i", video_tmp, "-i", wav_tmp,
                     "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                     "-shortest", final_path],
                    stderr=subprocess.PIPE,
                )
                if mux.returncode != 0:
                    tail = (mux.stderr[-2000:].decode("utf-8", "replace").strip()
                            if mux.stderr else "")
                    raise RuntimeError(
                        "[LTXAVStreamingSave] audio mux failed "
                        f"(ffmpeg exit {mux.returncode}); silent video "
                        f"kept at {video_tmp}"
                        + (f"\n--- ffmpeg stderr ---\n{tail}" if tail else "")
                    )
            finally:
                # wav_tmp is intermediate either way; video_tmp survives a mux
                # failure on purpose (it's the user's silent fallback).
                try:
                    os.remove(wav_tmp)
                except OSError:
                    pass
            os.remove(video_tmp)
        else:
            os.replace(video_tmp, final_path)

        print(f"[LTXAVStreamingSave] {frames_written} frames "
              f"({frames_written / fps:.2f}s) -> {final_path}")
        # Inline video preview (core SaveVideo convention). The player streams
        # the file from disk via /view — previewing costs no RAM at any length.
        return {
            "ui": {
                "images": [{
                    "filename": os.path.basename(final_path),
                    "subfolder": subfolder,
                    "type": "output",
                }],
                "animated": (True,),
            },
            "result": (final_path,),
        }


class LTXInpaintColorFill:
    """
    Composites a solid fill color where the mask is active — inpaint guide
    prep for IC-LoRAs that read the mask from the reference pixels. Color
    conventions differ per LoRA (Lightricks in/outpainting: #66FF00 green;
    community masked-inpaint LoRAs: magenta mask / chroma green fill), so the
    color is a preset choice with a custom hex fallback, unlike core's
    hardcoded LTXVInpaintPreprocess.

    Composite at the FINAL encode resolution (resize source and mask first) —
    resizing after compositing smears the fill boundary into off-colors the
    LoRA was never trained on. `binarize` (default on) thresholds the mask so
    the fill is exact even from soft/grown masks.
    """

    _PRESETS = {
        "magenta (255,0,255)":         (255, 0, 255),
        "chroma green (0,255,0)":      (0, 255, 0),
        "lightricks green (102,255,0)": (102, 255, 0),
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "mask": ("MASK", {
                    "tooltip": "White = filled with the color. Single-frame "
                               "masks broadcast to the video length.",
                }),
                "color": (list(s._PRESETS) + ["custom"], {
                    "default": "magenta (255,0,255)",
                }),
                "custom_hex": ("STRING", {
                    "default": "#FF00FF",
                    "tooltip": "Used when color = custom. #RRGGBB.",
                }),
                "binarize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Threshold the mask at 0.5 so the fill color is "
                               "exact (soft mask edges would blend off-colors).",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION     = "fill"
    CATEGORY     = "LTXAVTools/utils"
    DESCRIPTION  = (
        "Solid-color mask fill for inpaint IC-LoRA references (magenta / "
        "chroma green / Lightricks green / custom). Exact colors, unlike "
        "resize-after-composite pipelines."
    )

    def fill(self, images, mask, color, custom_hex, binarize):
        if color == "custom":
            h = custom_hex.strip().lstrip("#")
            if len(h) != 6:
                raise ValueError(
                    f"[LTXInpaintColorFill] custom_hex must be #RRGGBB, got "
                    f"{custom_hex!r}"
                )
            rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        else:
            rgb = self._PRESETS[color]

        m = mask
        if m.ndim == 4:
            m = m[:, :, :, 0]
        if binarize:
            m = (m > 0.5).float()
        if m.shape[0] == 1 and images.shape[0] > 1:
            m = m.expand(images.shape[0], -1, -1)
        n = min(m.shape[0], images.shape[0])
        if n < images.shape[0]:
            print(f"[LTXInpaintColorFill] mask has {m.shape[0]} frames, video "
                  f"{images.shape[0]} — output truncated to {n}.")
        m = m[:n].to(images.device, images.dtype)
        imgs = images[:n]
        if m.shape[1:] != imgs.shape[1:3]:
            m = torch.nn.functional.interpolate(
                m[:, None], size=imgs.shape[1:3], mode="nearest",
            )[:, 0]
            print(f"[LTXInpaintColorFill] mask resized to {imgs.shape[2]}x"
                  f"{imgs.shape[1]} (nearest — composite at final resolution "
                  f"to avoid this).")

        m4 = m.unsqueeze(-1)
        fill = torch.tensor(rgb, device=imgs.device, dtype=imgs.dtype) / 255.0
        out = imgs * (1 - m4) + fill.view(1, 1, 1, 3) * m4
        return (out,)


class LTXStreamingVideoEncode:
    """
    Chunked VAE encode straight from a video file — the full pixel tensor
    never exists. Constant RAM at any source length: frames are read from
    disk one chunk at a time, encoded with left pixel context, and only the
    (tiny) latents accumulate.

    Mirror of LTXAVStreamingSave's causal math: each chunk is encoded with
    `context_latents` of left context plus the 1-frame head pixel, and the
    context's latents (including the malformed 1-frame head latent) are
    trimmed from the output — the same trick that makes chunked decode
    exact, applied in reverse. Validate once per setup: encode a short clip
    both ways and compare (LTX AV Latent Check) before trusting long runs.

    Encodes FILES (e.g. Video Cut Marker's video_path). Branches that need
    in-graph preprocessing (DWPose/depth) should save the preprocessed video
    to disk first, then stream-encode that file.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to the source video file (wire from Video "
                               "Cut Marker's video_path or type directly).",
                }),
                "vae": ("VAE",),
                "width": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 32,
                    "tooltip": "Resize width before encoding (0 = native). "
                               "Snapped to ÷32. For small-grid IC guides use "
                               "gen/factor here.",
                }),
                "height": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 32,
                    "tooltip": "Resize height before encoding (0 = native, "
                               "snapped ÷32).",
                }),
                "chunk_latents": ("INT", {
                    "default": 16, "min": 2, "max": 256,
                    "tooltip": "Latent frames encoded per chunk (~ chunk*8 "
                               "pixel frames of RAM at a time).",
                }),
                "context_latents": ("INT", {
                    "default": 4, "min": 1, "max": 16,
                    "tooltip": "Left-context latents re-encoded with each chunk "
                               "and trimmed. Must cover the causal VAE encoder's "
                               "temporal receptive field; raise if the A/B "
                               "against a full encode ever shows a boundary "
                               "difference.",
                }),
                "force_rate": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 120.0, "step": 0.01,
                    "tooltip": "Resample to this fps while reading (0 = native "
                               "rate). Same accumulator scheme as VHS (24→25 "
                               "safe).",
                }),
                "frame_load_cap": ("INT", {
                    "default": 0, "min": 0, "max": 1_000_000,
                    "tooltip": "Max pixel frames to read after rate/skip "
                               "(0 = all). Wire from Video Cut Marker's "
                               "frame_load_cap.",
                }),
                "skip_first_frames": ("INT", {
                    "default": 0, "min": 0, "max": 1_000_000,
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "num_latents", "num_frames")
    FUNCTION     = "encode"
    CATEGORY     = "LTXAVTools/utils"
    DESCRIPTION  = (
        "Long-source encode without the RAM cliff: reads a video file in "
        "chunks and VAE-encodes each with causal left context, so only the "
        "latents accumulate — the full pixel tensor never exists. The input "
        "mirror of LTX AV Streaming Decode & Save."
    )

    def _frame_gen(self, path, force_rate, skip, cap_frames):
        import cv2
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"[LTXStreamingVideoEncode] could not open: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        base_t   = (1.0 / fps) if fps > 0 else 0.0
        target_t = (1.0 / force_rate) if force_rate > 0 else base_t
        time_acc = 0.0
        yielded = skipped = 0
        try:
            while True:
                if not cap.grab():
                    break
                if force_rate > 0 and fps > 0:
                    time_acc += base_t
                    if time_acc < target_t:
                        continue
                    time_acc -= target_t
                ok, frame = cap.retrieve()
                if not ok:
                    break
                if skipped < skip:
                    skipped += 1
                    continue
                yield frame  # BGR uint8 HWC
                yielded += 1
                if cap_frames and yielded >= cap_frames:
                    break
        finally:
            cap.release()

    def encode(self, video_path, vae, width, height, chunk_latents,
               context_latents, force_rate, frame_load_cap, skip_first_frames):
        import itertools
        import comfy.utils

        video_path = video_path.strip().strip('"')
        if not video_path or not os.path.isfile(video_path):
            raise ValueError(
                f"[LTXStreamingVideoEncode] video_path is not a file: "
                f"{video_path!r}"
            )

        gen = self._frame_gen(video_path, force_rate, skip_first_frames,
                              frame_load_cap)

        def take(n):
            return list(itertools.islice(gen, n))

        def to_tensor(frames):
            # BGR uint8 -> RGB float [T,H,W,C], resized to target dims
            arr = np.stack(frames)[..., ::-1]
            t = torch.from_numpy(np.ascontiguousarray(arr)).float() / 255.0
            if t.shape[1] != th or t.shape[2] != tw:
                t = comfy.utils.common_upscale(
                    t.movedim(-1, 1), tw, th, "lanczos", crop="center",
                ).movedim(1, -1).clamp(0, 1)
            return t

        # First chunk: n0 latents need 8*(n0-1)+1 pixel frames.
        first = take(8 * (chunk_latents - 1) + 1)
        if not first:
            raise ValueError(
                "[LTXStreamingVideoEncode] no frames read (empty video, or "
                "skip_first_frames past the end)."
            )
        # target dims: explicit, else native snapped to /32
        H0, W0 = first[0].shape[0], first[0].shape[1]
        th = height if height > 0 else max(32, int(round(H0 / 32)) * 32)
        tw = width  if width  > 0 else max(32, int(round(W0 / 32)) * 32)
        th, tw = max(32, (th // 32) * 32), max(32, (tw // 32) * 32)
        if (th, tw) != (H0, W0):
            print(f"[LTXStreamingVideoEncode] resizing {W0}x{H0} -> {tw}x{th}")

        drop = (len(first) - 1) % 8
        if drop:
            # short video: keep the valid (T-1)*8+1 head, note the trim
            print(f"[LTXStreamingVideoEncode] source ended mid-latent — "
                  f"dropping {drop} tail frame(s).")
            first = first[:len(first) - drop]

        chunks = []
        frames_read = len(first)
        window = to_tensor(first)
        del first
        lat = vae.encode(window)
        chunks.append(lat.cpu())
        total_latents = lat.shape[2]
        print(f"[LTXStreamingVideoEncode] frames [0,{frames_read}) -> "
              f"latents [0,{total_latents})")

        tail_len = 8 * context_latents + 1
        while True:
            model_management.throw_exception_if_processing_interrupted()
            new = take(8 * chunk_latents)
            if not new:
                break
            rem = len(new) % 8
            if rem:
                print(f"[LTXStreamingVideoEncode] source ended mid-latent — "
                      f"dropping {rem} tail frame(s).")
                new = new[:len(new) - rem]
                if not new:
                    break
            n = len(new) // 8
            # window = aligned suffix of the previous window (length ≡ 1 mod 8:
            # context latents' pixels + the 1-frame head) + the new frames. The
            # head latent and context latents are trimmed from the encode.
            avail = 8 * min(context_latents, (window.shape[0] - 1) // 8) + 1
            window = torch.cat([window[-avail:], to_tensor(new)], dim=0)
            start_f = frames_read
            frames_read += 8 * n
            del new
            lat = vae.encode(window)
            if lat.shape[2] < n:
                raise RuntimeError(
                    f"[LTXStreamingVideoEncode] encoder returned {lat.shape[2]} "
                    f"latents for a window expecting >= {n} — unexpected VAE "
                    f"temporal mapping."
                )
            chunks.append(lat[:, :, -n:].cpu())
            total_latents += n
            print(f"[LTXStreamingVideoEncode] frames [{start_f},{frames_read}) "
                  f"-> latents [{total_latents - n},{total_latents})")

        del window
        latent = torch.cat(chunks, dim=2) if len(chunks) > 1 else chunks[0]
        print(f"[LTXStreamingVideoEncode] done: {frames_read} frames -> "
              f"{latent.shape[2]} latents ({tw}x{th}).")
        return ({"samples": latent}, int(latent.shape[2]), int(frames_read))


class LTXVideoOutpaintLatent:
    """
    Latent-space outpaint prep for the base-model (no-LoRA) path. Zero-pads an
    encoded VIDEO latent spatially — real content in the center, ZEROS in the
    margin — and emits the matching feathered denoise mask.

    The zeros margin is the same empty substrate a from-scratch generation
    starts from, so the sampler noises and regenerates it per the schedule.
    This is the fix for "the model can't handle the padded pixels": encoding
    padded pixels bakes STRUCTURED content (encoded black/grey/green is a
    non-zero latent the model reads as "stuff is here" and tries to preserve),
    whereas a zero margin is nothing to preserve — pure generation target.

    Feed the padded latent (after concatenating audio) to the looping sampler's
    `latents`, and the mask to `optional_denoise_mask`. **Run in a FULL-denoise
    pass** so the margin actually regenerates — a low-denoise refinement won't
    add enough noise to a bare margin (that's what `margin_fill = noise` is for,
    experimental).

    Padding is in pixels, snapped to the LTX spatial grid (÷32). Feather ramps
    the mask INWARD into the original (blends the seam by partially regenerating
    the original's edge); feathering into the zeros margin would blend toward
    empty and muddy the seam, so it is deliberately one-directional.

    No LoRA, no color fill. This addresses the black-artifact failure, not the
    one-sided-context limit — strongest on moving-camera / simple margins.
    """

    VAE_SPATIAL = 32

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", {
                    "tooltip": "Encoded VIDEO latent (5D [B,C,T,H,W]) — not an AV "
                               "NestedTensor. Separate the video, outpaint it, then "
                               "re-concat audio.",
                }),
                "left":   ("INT", {"default": 0, "min": 0, "max": 8192, "step": 32}),
                "top":    ("INT", {"default": 0, "min": 0, "max": 8192, "step": 32}),
                "right":  ("INT", {"default": 0, "min": 0, "max": 8192, "step": 32}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 32}),
                "overlap": ("INT", {
                    "default": 0, "min": 0, "max": 512, "step": 8,
                    "tooltip": "Full-regenerate band (px) of the original adjacent "
                               "to the seam — fully rewritten (mask=1) so the seam "
                               "sits inside one continuous generation, not on a "
                               "partial-keep boundary. Then `feather` ramps to kept. "
                               "Keep it small (~16-32); it discards that strip of "
                               "real content. 0 = feather starts at the seam.",
                }),
                "feather": ("INT", {
                    "default": 32, "min": 0, "max": 512, "step": 8,
                    "tooltip": "Mask feather (px) INTO the original, beyond the "
                               "overlap band, ramping regen->keep. 0 = hard edge.",
                }),
            },
            "optional": {
                "margin_fill": (["zeros", "noise"], {
                    "default": "zeros",
                    "tooltip": "zeros = correct for a full-denoise pass (the sampler "
                               "noises it). noise = pre-populate the margin with unit "
                               "noise for low-denoise passes (experimental — the "
                               "sampler still adds its own noise on top).",
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "MASK")
    RETURN_NAMES = ("latent", "denoise_mask")
    FUNCTION     = "outpaint"
    CATEGORY     = "LTXAVTools/utils"
    DESCRIPTION  = (
        "Zero-pads an encoded video latent for base-model outpaint (real centre, "
        "empty margin) and emits the feathered denoise mask. Margin regenerates "
        "cleanly because it is empty, not structured padded pixels. No LoRA."
    )

    def outpaint(self, samples, left, top, right, bottom, feather,
                 overlap=0, margin_fill="zeros"):
        v = samples["samples"]
        if _HAS_NESTED and isinstance(v, NestedTensor):
            raise ValueError(
                "[LTXVideoOutpaintLatent] got an AV NestedTensor — separate the "
                "video latent first (LTXVSeparateAVLatent), outpaint it, then "
                "re-concat audio."
            )
        if v.ndim != 5:
            raise ValueError(
                f"[LTXVideoOutpaintLatent] expected a 5D video latent [B,C,T,H,W], "
                f"got {v.ndim}D."
            )
        S = self.VAE_SPATIAL

        def snap(p, name):
            q = max(0, int(round(p / S)) * S)
            if q != p:
                print(f"[LTXVideoOutpaintLatent] {name} {p} snapped to {q} (÷{S}).")
            return q

        left, top, right, bottom = (snap(left, "left"), snap(top, "top"),
                                    snap(right, "right"), snap(bottom, "bottom"))
        Lc, Tc, Rc, Bc = left // S, top // S, right // S, bottom // S
        B, C, T, H, W = v.shape

        # zero-pad the latent spatially (F.pad's last-dims-first order: W then H)
        v_pad = torch.nn.functional.pad(v, (Lc, Rc, Tc, Bc), value=0.0)

        if margin_fill == "noise":
            gen = torch.randn_like(v_pad)
            keep = torch.zeros((1, 1, 1, v_pad.shape[3], v_pad.shape[4]),
                               dtype=v_pad.dtype, device=v_pad.device)
            keep[..., Tc:Tc + H, Lc:Lc + W] = 1.0
            v_pad = torch.where(keep.bool(), v_pad, gen)

        # feathered denoise mask at padded PIXEL resolution (single frame)
        H_out, W_out = (Tc + Bc + H) * S, (Lc + Rc + W) * S
        y0, x0 = Tc * S, Lc * S
        y1, x1 = y0 + H * S, x0 + W * S
        dev = v.device
        yy = torch.arange(H_out, device=dev).view(-1, 1).float()
        xx = torch.arange(W_out, device=dev).view(1, -1).float()
        BIG = float(max(H_out, W_out) + 1)
        dt = (yy - y0)      if top    > 0 else torch.full((H_out, 1), BIG, device=dev)
        db = (y1 - 1 - yy)  if bottom > 0 else torch.full((H_out, 1), BIG, device=dev)
        dl = (xx - x0)      if left   > 0 else torch.full((1, W_out), BIG, device=dev)
        dr = (x1 - 1 - xx)  if right  > 0 else torch.full((1, W_out), BIG, device=dev)
        d = torch.minimum(torch.minimum(dt, db), torch.minimum(dl, dr))  # -> [H_out,W_out]
        in_orig = ((yy >= y0) & (yy < y1)) & ((xx >= x0) & (xx < x1))
        # d = distance into the original from the seam. [0, overlap) fully
        # regenerates (mask 1); [overlap, overlap+feather) ramps 1->0; beyond
        # keeps (mask 0).
        if feather > 0:
            orig_vals = ((overlap + feather - d) / feather).clamp(0.0, 1.0)
        else:
            orig_vals = (d < overlap).float()
        mask = torch.where(in_orig, orig_vals.expand(H_out, W_out),
                           torch.ones(H_out, W_out, device=dev)).unsqueeze(0)

        print(f"[LTXVideoOutpaintLatent] {W}x{H} latent -> {v_pad.shape[4]}x"
              f"{v_pad.shape[3]} (pad cells L{Lc} T{Tc} R{Rc} B{Bc}); "
              f"margin={margin_fill}, overlap {overlap}px, feather {feather}px.")
        return ({"samples": v_pad}, mask)


NODE_CLASS_MAPPINGS = {
    "PreviewImagePassthrough":          PreviewImagePassthrough,
    "LTXAVLatentCheck":                 LTXAVLatentCheck,
    "LTXAVSeparateCheck":               LTXAVSeparateCheck,
    "LTXAudioLatentPad":                LTXAudioLatentPad,
    "LTXVAVLatentUpsampler":            LTXVAVLatentUpsampler,
    "LTXVAVLatentUpsamplerTiled":       LTXVAVLatentUpsamplerTiled,
    "LTXKeyframePairConcat":            LTXKeyframePairConcat,
    "LTXLoraMetadataReader":            LTXLoraMetadataReader,
    "LTXAVStreamingSave":               LTXAVStreamingSave,
    "LTXStreamingVideoEncode":          LTXStreamingVideoEncode,
    "LTXInpaintColorFill":              LTXInpaintColorFill,
    "LTXVideoOutpaintLatent":           LTXVideoOutpaintLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PreviewImagePassthrough":          "Preview Image Passthrough",
    "LTXAVLatentCheck":                 "LTX AV Latent Check",
    "LTXAVSeparateCheck":               "LTX AV Separate Check",
    "LTXAudioLatentPad":                "LTX Audio Latent Pad",
    "LTXVAVLatentUpsampler":            "LTX AV Latent Upsampler",
    "LTXVAVLatentUpsamplerTiled":       "LTX AV Latent Upsampler (Tiled)",
    "LTXKeyframePairConcat":            "LTX Keyframe Pair Concat",
    "LTXLoraMetadataReader":            "LTX LoRA Metadata Reader",
    "LTXAVStreamingSave":               "LTX AV Streaming Decode & Save",
    "LTXStreamingVideoEncode":          "LTX Streaming Video Encode",
    "LTXInpaintColorFill":              "LTX Inpaint Color Fill",
    "LTXVideoOutpaintLatent":           "LTX Video Outpaint Latent",
}
