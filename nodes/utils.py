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

        out_dir = folder_paths.get_output_directory()
        full_folder, fname, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, out_dir
        )
        video_tmp  = os.path.join(full_folder, f"{fname}_{counter:05}_tmp.mp4")
        final_path = os.path.join(full_folder, f"{fname}_{counter:05}.mp4")

        proc = None
        frames_written = 0
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
                    )

                data = (
                    px.clamp(0, 1).mul(255).round()
                      .to(torch.uint8).cpu().contiguous().numpy().tobytes()
                )
                proc.stdin.write(data)
                frames_written += px.shape[0]
                print(f"[LTXAVStreamingSave] latents [{k},{k + n}) of {T} -> "
                      f"{px.shape[0]} frames (total {frames_written})")
                del px, data
                k += n

            proc.stdin.close()
            ret = proc.wait()
            if ret != 0:
                raise RuntimeError(f"[LTXAVStreamingSave] ffmpeg exited with {ret}.")
            proc = None
        finally:
            if proc is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                proc.kill()

        if optional_audio is not None and optional_audio.get("waveform") is not None:
            import torchaudio
            wav_tmp = os.path.join(full_folder, f"{fname}_{counter:05}_tmp.wav")
            wf = optional_audio["waveform"]
            if wf.ndim == 3:
                wf = wf[0]
            torchaudio.save(wav_tmp, wf.cpu(), int(optional_audio["sample_rate"]))
            mux = subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error",
                 "-i", video_tmp, "-i", wav_tmp,
                 "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                 "-shortest", final_path],
            )
            if mux.returncode != 0:
                raise RuntimeError("[LTXAVStreamingSave] audio mux failed "
                                   f"(ffmpeg exit {mux.returncode}); silent video "
                                   f"kept at {video_tmp}")
            os.remove(video_tmp)
            os.remove(wav_tmp)
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
}
