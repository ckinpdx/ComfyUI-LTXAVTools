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


AUDIO_LATENTS_PER_SECOND = 25.0


def audio_pos(n_latents, fps):
    """Audio latents covering video latents [0, n) — the GLOBAL boundary map.

    Every audio count should be a DIFFERENCE of two audio_pos values, never a
    chunk-local span. Closed form, for q = 200/fps and n >= 1:

        audio_pos(n) = (n - 1) * q + floor(q/8 + 0.5)

    When q is an integer (fps divides 200), (n-1)*q factors straight out of the
    rounding, so the rounding term is a CONSTANT and cancels in any difference:

        audio_pos(n) - audio_pos(m) = (n - m) * q     exactly, for n > m >= 1

    A chunk-LOCAL span instead re-introduces LTX's first-frame asymmetry
    (latent 0 = 1 pixel frame) at every chunk; at 50 fps that lands on exactly
    x.5 and rounds by parity, giving +-1 audio frame per boundary, cumulative.
    See SPEC_50FPS.md.

    Half-up rounding — never Python round(), whose banker's rounding is
    parity-dependent. Exact in float64 for every valid fps: fps | 200 = 2^3*5^2
    forces 25/fps = 5^(2-b)/2^a, a dyadic rational.
    """
    if n_latents <= 0:
        return 0
    px = (n_latents - 1) * 8 + 1
    return int(px * AUDIO_LATENTS_PER_SECOND / fps + 0.5)


# ORDER IS PREFERENCE. Short clips can fit more than one rate (23.976 vs 24,
# 48 vs 50), and the first match wins — so the rates people actually work at
# come first and the oddballs last. Without this a 24 fps clip resolves to
# 23.976 and every downstream duration inherits the drift.
FPS_CANDIDATES = (25, 24, 50, 30, 60, 48, 20, 40, 15, 12, 10, 8, 5, 4, 2, 1,
                  100, 120, 200, 23.976)


def infer_av_fps(T_v, T_a, candidates=FPS_CANDIDATES):
    """Recover a clip's frame rate from its own video:audio latent ratio.

    Audio runs at a fixed 25 latents/second regardless of video fps, so
    `audio_pos(T_v, fps) == T_a` pins the rate — a clip can be asked what it is
    instead of trusted. Returns every candidate that fits (normally one; short
    clips can be ambiguous, hence a list).
    """
    if T_v < 2:
        return []                       # a 1-latent clip carries no ratio
    return [f for f in candidates if audio_pos(T_v, f) == T_a]


def ltx_video_mask_to_audio_profile(mask_5d, fps):
    """[1,1,T,h,w] video denoise mask -> [audio_pos(T, fps)] audio profile.

    Derived rather than authored, so the two modalities cannot disagree about
    where a frozen span is. Video latent t owns audio [audio_pos(t),
    audio_pos(t+1)) — with audio_pos(0) taken as 0, latent 0 owns exactly one
    audio frame and every later latent owns q = 200/fps. Spatial reduction is
    MAX: if any part of a video frame is being regenerated, its audio is too
    (freezing audio under a partially-regenerating frame would fight the video).
    """
    T = int(mask_5d.shape[2])
    per_frame = mask_5d.amax(dim=(3, 4)).reshape(-1)          # [T]
    total = audio_pos(T, fps)
    prof = torch.ones(total, device=mask_5d.device, dtype=mask_5d.dtype)
    for t in range(T):
        s = 0 if t == 0 else audio_pos(t, fps)
        e = min(total, audio_pos(t + 1, fps))
        if e > s:
            prof[s:e] = per_frame[t]
    return prof


def ltx_mask_to_latent(m, T, lat_h, lat_w, mode="max"):
    """Pixel MASK -> [1,1,T,lat_h,lat_w] on the LTX latent grid.

    Each latent frame covers a group of pixel frames (LTX first-frame asymmetry:
    latent 0 = 1 pixel frame, latents 1+ = 8). `mode` = how those frames reduce:
      - max            : union — covers the object wherever it appears in the
                         group (safe, but oversizes on motion; blobs at cuts).
      - min            : intersection — only where masked in EVERY frame (crisp,
                         under-covers on motion -> can leak).
      - last           : the group's LAST pixel frame only — the LTX VAE is
                         causal, so latent t aligns to pixel frame 8t. Crisp,
                         tracks the current position, no union blob.
      - mean_threshold : average then threshold at 0.5 — middle ground.
    Spatial = bilinear."""
    F = torch.nn.functional
    if m.ndim == 2:
        m = m.unsqueeze(0)
    elif m.ndim == 4:
        m = m.squeeze(1) if m.shape[1] == 1 else m[0]
    m = m.float().clamp(0.0, 1.0)                                       # [N,H,W]
    # Spatial downsample by MAX, not bilinear: a latent cell must be masked if
    # the subject touches it AT ALL. Bilinear point-samples a 32x reduction, so
    # the whole boundary ring reads ~0 (silently treated as KEEP -> pinned to
    # the init, which decodes as a grey ring after a noise-fill) and thin
    # features come out fractional (mask < 1 re-blends the init every step, see
    # comfy samplers.py CFGGuider.__call__). Max matches the temporal reduction.
    ms = F.adaptive_max_pool2d(m.unsqueeze(1), (lat_h, lat_w)).squeeze(1)  # [N,lh,lw]
    N = ms.shape[0]
    if N == 1:
        out = ms.expand(T, -1, -1)
    elif N == T:
        out = ms
    else:
        px = (T - 1) * 8 + 1
        if N != px:
            # normalize odd frame counts to the pixel grid without averaging
            ms = F.interpolate(ms[None, None], size=(px, lat_h, lat_w),
                               mode="nearest")[0, 0]

        def _reduce(grp):  # grp: [k, lh, lw]
            if mode == "min":
                return grp.amin(dim=0, keepdim=True)
            if mode == "last":
                return grp[-1:].clone()
            if mode == "mean_threshold":
                return (grp.mean(dim=0, keepdim=True) > 0.5).float()
            return grp.amax(dim=0, keepdim=True)  # max (default)

        groups = [ms[0:1]]
        for t in range(1, T):
            groups.append(_reduce(ms[8 * (t - 1) + 1: 8 * t + 1]))
        out = torch.cat(groups, dim=0)                                 # [T,lh,lw]
    return out[None, None].clamp(0.0, 1.0)                             # [1,1,T,lh,lw]


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

    AUDIO_LATENTS_PER_SECOND = AUDIO_LATENTS_PER_SECOND  # module constant

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

        # SPEC_50FPS: global boundary map, not a local span (the old
        # round(px/fps*25) form reports false mismatches at 50 fps).
        expected = audio_pos(T_v, fps)
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

    AUDIO_LATENTS_PER_SECOND = AUDIO_LATENTS_PER_SECOND  # module constant

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

        # SPEC_50FPS: global boundary map, not a local span (the old
        # round(px/fps*25) form reports false mismatches at 50 fps).
        expected = audio_pos(T_v, fps)
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

    # APPEND-ONLY: `factor_int` is deliberately LAST. ComfyUI links outputs by
    # INDEX, so slotting it in beside the FLOAT would silently re-point every
    # existing `metadata` link in saved workflows.
    RETURN_TYPES = ("STRING", "FLOAT", "STRING", "INT")
    RETURN_NAMES = ("lora_path", "latent_downscale_factor", "metadata", "factor_int")
    FUNCTION     = "read"
    CATEGORY     = "LTXAVTools/utils"
    DESCRIPTION  = (
        "Reads a LoRA's safetensors metadata header (no weight loading). Outputs "
        "the absolute path (wire to a loader's opt_lora_path so one combo drives "
        "everything), the IC-LoRA reference_downscale_factor as both FLOAT and INT, "
        "and the full metadata for inspection. Use the FLOAT for the AV Looping "
        "Sampler's guiding_downscale_factor and the IC References node; use "
        "factor_int for INT-typed consumers such as LTXV Dilate Latent's "
        "horizontal_scale / vertical_scale, which will not accept a FLOAT link."
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

        # Key lookup is prefix-agnostic: Lightricks' own IC-LoRAs write
        # `reference_downscale_factor`, while LoRAs trained through
        # sd-scripts/kohya/musubi prefix training metadata with `ss_`
        # (`ss_reference_downscale_factor`). Match the bare key first, then any
        # key ending in it, so future prefixes work without another edit.
        raw, src_key = None, None
        for k in ("reference_downscale_factor", "ss_reference_downscale_factor"):
            if k in md:
                raw, src_key = md[k], k
                break
        if raw is None:
            for k in md:
                if k.endswith("reference_downscale_factor"):
                    raw, src_key = md[k], k
                    break

        try:
            factor = max(1.0, float(raw if raw is not None else 1))
        except (TypeError, ValueError):
            factor = 1.0

        meta_str = json.dumps(md, indent=2) if md else "(no metadata)"
        found = f"{src_key}={factor}" if src_key else f"no factor key (default {factor})"
        print(f"[LTXLoraMetadataReader] {lora_name}: {found} | "
              f"{len(md)} metadata keys")
        return (path, factor, meta_str, int(round(factor)))


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
                "video_encoder": (["auto", "libx264", "h264_nvenc",
                                   "libopenh264", "mpeg4"], {
                    "default": "auto",
                    "tooltip": "auto = probe the binary and take the first "
                               "available in preference order (libx264 → "
                               "h264_nvenc → libopenh264 → mpeg4). Pick one "
                               "explicitly if auto guesses wrong. Quality maps "
                               "per encoder: crf (x264) / cq (nvenc) / bitrate "
                               "(openh264) / qscale (mpeg4).",
                }),
                "ffmpeg_path": ("STRING", {
                    "default": "",
                    "tooltip": "Override the ffmpeg binary. Empty = search the "
                               "explicit path, then PATH, then imageio-ffmpeg, "
                               "and use the first one that actually has a "
                               "working H.264 encoder (a PATH ffmpeg built "
                               "without libx264 is the usual culprit).",
                }),
            },
        }

    # --- ffmpeg capability handling -------------------------------------
    # Each entry: (encoder name, builder(crf, w, h) -> arg list). -preset and
    # -crf are x264-specific, so every encoder needs its own quality mapping.
    @staticmethod
    def _encoder_args(name, crf, w, h):
        if name == "libx264":
            return ["-c:v", "libx264", "-preset", "medium",
                    "-crf", str(crf), "-pix_fmt", "yuv420p"]
        if name == "h264_nvenc":
            return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr",
                    "-cq", str(crf), "-pix_fmt", "yuv420p"]
        if name == "libopenh264":
            # no CRF support — approximate from crf and frame area
            mbps = max(1.0, (w * h / (1280 * 720)) * (2.0 ** ((28 - crf) / 6.0)))
            return ["-c:v", "libopenh264", "-b:v", f"{mbps:.1f}M",
                    "-pix_fmt", "yuv420p"]
        # mpeg4: qscale 1(best)-31(worst)
        return ["-c:v", "mpeg4", "-q:v", str(max(1, min(31, crf // 2))),
                "-pix_fmt", "yuv420p"]

    _ENCODER_PREFERENCE = ["libx264", "h264_nvenc", "libopenh264", "mpeg4"]
    _probe_cache = {}

    @classmethod
    def _available_encoders(cls, ffmpeg):
        """Encoders this binary can actually use (cached per binary path)."""
        if ffmpeg in cls._probe_cache:
            return cls._probe_cache[ffmpeg]
        import subprocess
        found = set()
        try:
            out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                                 capture_output=True, timeout=30)
            text = (out.stdout or b"").decode("utf-8", "replace")
            for enc in cls._ENCODER_PREFERENCE:
                # lines look like " V....D libx264   libx264 H.264 ..."
                if any(line.split()[1:2] == [enc]
                       for line in text.splitlines() if line.strip()):
                    found.add(enc)
        except Exception as e:
            # The binary could not be run at all (missing / not executable).
            # Report NO encoders so this candidate is skipped rather than
            # optimistically assumed good — otherwise a bad explicit path or a
            # stale PATH entry shadows a working build.
            print(f"[LTXAVStreamingSave] ffmpeg candidate unusable, skipping: "
                  f"{ffmpeg} ({e})")
            found = set()
        cls._probe_cache[ffmpeg] = found
        return found

    @classmethod
    def _resolve_ffmpeg(cls, explicit, want_encoder):
        """Pick (binary, encoder). Prefers a binary that HAS a usable encoder
        over merely the first one found — a PATH ffmpeg built without libx264
        (conda) would otherwise shadow a working imageio-ffmpeg build."""
        import shutil
        candidates = []
        if explicit and explicit.strip():
            candidates.append(explicit.strip().strip('"'))
        on_path = shutil.which("ffmpeg")
        if on_path:
            candidates.append(on_path)
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            candidates.append(get_ffmpeg_exe())
        except Exception:
            pass
        # de-dupe, keep order
        seen, ordered = set(), []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                ordered.append(c)
        if not ordered:
            raise RuntimeError(
                "[LTXAVStreamingSave] no ffmpeg found (not on PATH, "
                "imageio-ffmpeg unavailable, no ffmpeg_path given)."
            )

        wanted = ([want_encoder] if want_encoder != "auto"
                  else cls._ENCODER_PREFERENCE)
        for binary in ordered:
            have = cls._available_encoders(binary)
            for enc in wanted:
                if enc in have:
                    return binary, enc
        # nothing matched: fall back to the first binary and let ffmpeg speak
        fallback = wanted[0]
        print(f"[LTXAVStreamingSave] no candidate ffmpeg reported {wanted} — "
              f"trying {ordered[0]} with {fallback} anyway.")
        return ordered[0], fallback

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
                    filename_prefix, crf, optional_audio=None,
                    video_encoder="auto", ffmpeg_path=""):
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

        # Pick a binary that actually HAS a usable encoder, not just the first
        # one on PATH — logged so shadowed-PATH problems are visible in reports.
        ffmpeg, encoder = self._resolve_ffmpeg(ffmpeg_path, video_encoder)
        print(f"[LTXAVStreamingSave] using ffmpeg: {ffmpeg} | encoder: {encoder}"
              + ("" if video_encoder == "auto" else " (forced)"))

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
                    if (W % 2) or (H % 2):
                        raise ValueError(
                            f"[LTXAVStreamingSave] frame size {W}x{H} has an odd "
                            f"dimension; yuv420p requires both even. LTX latents "
                            f"are always ÷32 — a hand-cropped latent is the usual "
                            f"cause."
                        )
                    proc = subprocess.Popen(
                        [ffmpeg, "-y", "-loglevel", "error",
                         "-f", "rawvideo", "-pix_fmt", "rgb24",
                         "-s", f"{W}x{H}", "-r", str(fps), "-i", "pipe:"]
                        + self._encoder_args(encoder, crf, W, H)
                        + [video_tmp],
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
                m[:, None], size=imgs.shape[1:3], mode="bilinear",
                align_corners=False,
            )[:, 0]
            print(f"[LTXInpaintColorFill] mask resized to {imgs.shape[2]}x"
                  f"{imgs.shape[1]} — source-grid blocks would be "
                  f"{imgs.shape[2] / m.shape[2]:.2f}px wide with nearest; "
                  f"composite at final resolution to avoid resampling at all.")

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
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        out_fps = force_rate if force_rate > 0 else native_fps
        try:
            # No resample (rates equal or unknown): 1:1 passthrough.
            if native_fps <= 0 or out_fps <= 0 or abs(out_fps - native_fps) < 1e-6:
                skipped = emitted = 0
                while True:
                    if not cap.grab():
                        break
                    ok, frame = cap.retrieve()
                    if not ok:
                        break
                    if skipped < skip:
                        skipped += 1
                        continue
                    yield frame  # BGR uint8 HWC
                    emitted += 1
                    if cap_frames and emitted >= cap_frames:
                        break
                return

            # Nearest/hold resample so OUTPUT-frame indices match a VHS
            # force_rate stream: duplicates when upsampling (e.g. 24->25),
            # drops when downsampling. skip / cap_frames count OUTPUT frames —
            # the same emit_fps space the Cut Marker emits, so its
            # skip_first_frames / frame_load_cap land on the same content the
            # VHS loader sees. Output frame j sources native frame
            # floor(j * native_fps / out_fps); the index is recomputed per j
            # (no float accumulation) so it can't drift over long videos.
            in_idx = -1
            out_idx = 0
            skipped = emitted = 0
            frame = None
            while True:
                need = int(out_idx * native_fps / out_fps)
                while in_idx < need:
                    if not cap.grab():
                        return
                    ok, frame = cap.retrieve()
                    if not ok:
                        return
                    in_idx += 1
                out_idx += 1
                if frame is None:
                    continue
                if skipped < skip:
                    skipped += 1
                    continue
                yield frame
                emitted += 1
                if cap_frames and emitted >= cap_frames:
                    return
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


class LTXNoiseFill:
    """
    Pixel-space removal fill: composites noise into the masked region of a
    video BEFORE encoding. Removing the subject in *pixels* (not latent cells)
    means it's gone from the latent everywhere — so the VAE can't smear it from
    kept cells back into the hole, which is what leaks with latent-space
    zeroing. Noise (vs a solid fill) is unstructured, so the encoder spreads
    noise, not content, and the model regenerates the hole fresh instead of
    anchoring to prior pixels.

    `noise_mode`:
      - decoded: a random latent run through the VAE decode → on-manifold noise
        pixels that re-encode to clean latent noise (one VAE decode; the cost).
      - gaussian: cheap per-frame pixel noise, no decode (use if decoded is too
        heavy — slightly off-manifold for the encoder but usually fine).

    Feed the output IMAGE to your normal VAE encode → sampler `latents`, and the
    returned MASK to optional_denoise_mask. HARD mask + small grow (no feather)
    so the hole covers the encoder's ~32px spread. Everything runs on the GPU.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "mask": ("MASK",),
                "vae": ("VAE",),
                "noise_mode": (["decoded", "gaussian"], {"default": "decoded"}),
            },
            "optional": {
                "grow": ("INT", {
                    "default": 0, "min": 0, "max": 256, "step": 1,
                    "tooltip": "Keep at 0 — the removal should be TIGHT. Grow "
                               "belongs on the DENOISE mask instead (GrowMaskWithBlur "
                               "right before the sampler): grow+blur the regenerate "
                               "region to cover the VAE's ~32px smear and blend the "
                               "seam, while the fill stays precise. Separable GPU dilate "
                               "if you do use it.",
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "noise_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1,
                    "tooltip": "Std of the random latent (decoded mode) — 1.0 ~ "
                               "encoded-content scale.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION     = "fill"
    CATEGORY     = "LTXAVTools/utils"
    DESCRIPTION  = (
        "Composite noise into a masked region in PIXEL space before encoding, "
        "so the subject is gone from the latent entirely (no VAE smear/leak). "
        "The correct removal prep — wire IMAGE → your VAE encode, MASK → "
        "optional_denoise_mask."
    )

    def fill(self, images, mask, vae, noise_mode="decoded",
             grow=32, seed=0, noise_scale=1.0, _label="LTXNoiseFill"):
        # _label lets a wrapping node (LTXRemovalEncode) print under its OWN
        # name — a console line from a node that is not on the canvas is
        # actively confusing when you are trying to trace a mask problem.
        F = torch.nn.functional
        dev = model_management.get_torch_device()
        imgs = images.to(dev)                                # [N,H,W,3]
        N, H, W, _ = imgs.shape

        # mask -> [N,H,W] on GPU, spatial-matched, grown, hard
        m = mask.to(dev).float()
        if m.ndim == 2:
            m = m.unsqueeze(0)
        elif m.ndim == 4:
            m = m.squeeze(1) if m.shape[1] == 1 else m[0]
        m = m.clamp(0.0, 1.0)
        if m.shape[0] == 1 and N > 1:
            m = m.expand(N, -1, -1)
        if m.shape[1] != H or m.shape[2] != W:
            # Report it: a mask arriving at a different resolution than the video
            # is resampled here, and that is invisible in the graph. Nearest would
            # quantise the boundary to the SOURCE mask's grid — a 512-wide mask on
            # 1728-wide video gives 3.4 px blocks — so resample smoothly and
            # re-harden below. Coverage is unchanged; the edge follows the shape.
            print(f"[{_label}] mask is {m.shape[2]}x{m.shape[1]} but the video "
                  f"is {W}x{H} — resampling. Source-grid blocks would be "
                  f"{W / m.shape[2]:.2f}px wide.")
            m = F.interpolate(m.unsqueeze(1), size=(H, W), mode="bilinear",
                              align_corners=False).squeeze(1)
        else:
            print(f"[{_label}] mask matches the video at {W}x{H} — no resample.")
        if grow > 0:
            k = grow * 2 + 1
            mm = m.unsqueeze(1)
            mm = F.max_pool2d(mm, (k, 1), stride=1, padding=(grow, 0))
            mm = F.max_pool2d(mm, (1, k), stride=1, padding=(0, grow))
            m = mm.squeeze(1)
        m = (m > 0.5).float()                                # hard

        # noise pixels [N,H,W,3] on GPU
        g = torch.Generator(device=dev).manual_seed(int(seed))
        if noise_mode == "decoded":
            lat_c = getattr(vae, "latent_channels", 128)
            T_lat = (N - 1) // 8 + 1
            H_lat = max(1, round(H / 32))
            W_lat = max(1, round(W / 32))
            z = torch.randn(1, lat_c, T_lat, H_lat, W_lat,
                            generator=g, device=dev) * noise_scale
            noise_px = vae.decode(z)
            if isinstance(noise_px, tuple):
                noise_px = noise_px[0]
            if noise_px.ndim == 5:
                noise_px = noise_px.reshape(-1, *noise_px.shape[-3:])  # [frames,H,W,3]
            noise_px = noise_px.to(dev).clamp(0.0, 1.0)[:N]
            if noise_px.shape[1] != H or noise_px.shape[2] != W:
                noise_px = F.interpolate(noise_px.movedim(-1, 1), size=(H, W),
                                         mode="bilinear", align_corners=False).movedim(1, -1)
            if noise_px.shape[0] < N:  # short decode safety
                reps = -(-N // noise_px.shape[0])
                noise_px = noise_px.repeat(reps, 1, 1, 1)[:N]
        else:  # gaussian
            noise_px = (0.5 + 0.18 * torch.randn(N, H, W, 3, generator=g, device=dev)).clamp(0.0, 1.0)

        m4 = m.unsqueeze(-1)
        out = imgs * (1.0 - m4) + noise_px * m4
        print(f"[LTXNoiseFill] {W}x{H}x{N}: filled {float(m.mean()) * 100:.1f}% with "
              f"{noise_mode} noise (grow {grow}px). Encode this, mask -> "
              f"optional_denoise_mask.")
        return (out, m)


class LTXInpaintLatent:
    """
    Interior counterpart to LTXVideoOutpaintLatent: zeros the MASKED region of
    an encoded video latent (real content outside, EMPTY inside) and emits the
    matching denoise mask. Same empty-latent principle as outpaint — a zero
    region is nothing to preserve, so the masked area regenerates cleanly from
    the surrounding scene + prompt, with none of the structured 'ghost' of the
    removed content that masking over real latent can leave.

    Use for REMOVAL / full replacement (no trace of the original wanted). For
    in-place edits (recolor / restyle, where the original structure is the base
    you're modifying) feed the real latent + optional_denoise_mask instead —
    zeroing throws away the structure you're editing.

    Zeroing is proportional to the (grown/feathered) mask — latent × (1 − mask)
    — so the feather boundary fades empty→real in step with the denoise mask.
    Wire latent → sampler `latents` (after concatenating audio), denoise_mask →
    optional_denoise_mask.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", {
                    "tooltip": "Encoded VIDEO latent (5D [B,C,T,H,W]) — not an AV "
                               "NestedTensor. Separate video, inpaint, re-concat audio.",
                }),
                "mask": ("MASK", {
                    "tooltip": "White = remove/regenerate, black = keep. Single frame "
                               "is static; a batch (e.g. SAM per-frame) is resampled "
                               "onto the latent grid.",
                }),
            },
            "optional": {
                "grow": ("INT", {
                    "default": 0, "min": 0, "max": 256, "step": 1,
                    "tooltip": "Dilate the mask (px) — expand the removal region to "
                               "cover the object's edge, contact shadow, etc.",
                }),
                "feather": ("INT", {
                    "default": 16, "min": 0, "max": 256, "step": 1,
                    "tooltip": "Soften the mask boundary (px) so the zeroing and the "
                               "denoise fade empty→real across it. Applied to the same "
                               "mask used for both, so they stay consistent.",
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert: keep the masked region, regenerate the rest.",
                }),
                "temporal_mode": (["max", "last", "mean_threshold", "min"], {
                    "default": "max",
                    "tooltip": "How the per-frame mask reduces onto the 8:1 latent "
                               "grid. max = union (safe, blobs at cuts/motion); last "
                               "= the group's last frame (causal-aligned, crisp); "
                               "mean_threshold = middle; min = intersection. Try "
                               "'last' if you get gray blobs at view changes.",
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "MASK")
    RETURN_NAMES = ("latent", "denoise_mask")
    FUNCTION     = "inpaint"
    CATEGORY     = "LTXAVTools/utils"
    DESCRIPTION  = (
        "Zeros a masked region of an encoded video latent (empty inside, real "
        "outside) + emits the feathered denoise mask, for base-model removal "
        "inpaint. The empty region regenerates without a ghost of the removed "
        "content. No LoRA. Interior mirror of LTX Video Outpaint Latent."
    )

    @staticmethod
    def _gaussian_blur(m, radius):
        sigma = max(0.5, radius / 2.0)
        x = torch.arange(radius * 2 + 1, dtype=torch.float32) - radius
        k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        k = (k / k.sum()).to(m.device, m.dtype)
        mm = m.unsqueeze(1)  # [N,1,H,W]
        mm = torch.nn.functional.conv2d(mm, k.view(1, 1, 1, -1), padding=(0, radius))
        mm = torch.nn.functional.conv2d(mm, k.view(1, 1, -1, 1), padding=(radius, 0))
        return mm.squeeze(1).clamp(0.0, 1.0)

    def inpaint(self, samples, mask, grow=0, feather=16, invert=False,
                temporal_mode="max"):
        v = samples["samples"]
        if _HAS_NESTED and isinstance(v, NestedTensor):
            raise ValueError(
                "[LTXInpaintLatent] got an AV NestedTensor — separate the video "
                "latent first (LTXVSeparateAVLatent), inpaint it, re-concat audio."
            )
        if v.ndim != 5:
            raise ValueError(
                f"[LTXInpaintLatent] expected a 5D video latent [B,C,T,H,W], "
                f"got {v.ndim}D."
            )
        B, C, T, H, W = v.shape

        F = torch.nn.functional
        m = mask.float()
        if m.ndim == 2:
            m = m.unsqueeze(0)                                   # [1,H,W]
        elif m.ndim == 4:
            m = m.squeeze(1) if m.shape[1] == 1 else m[0]        # -> [N,H,W]
        # Morphology on the latent's device (GPU) — a full-res per-frame video
        # mask is far too heavy for CPU. Kept as [N,H,W]; ops are batched.
        m = m.clamp(0.0, 1.0).to(v.device)
        if invert:
            m = 1.0 - m
        if grow > 0:
            # Separable dilation: max over rows then columns (identical to a
            # square max-pool, ~kernel× cheaper — the non-separable version
            # hangs at grow 32).
            k = grow * 2 + 1
            mm = m.unsqueeze(1)                                  # [N,1,H,W]
            mm = F.max_pool2d(mm, (k, 1), stride=1, padding=(grow, 0))
            mm = F.max_pool2d(mm, (1, k), stride=1, padding=(0, grow))
            m = mm.squeeze(1)
        if feather > 0:
            m = self._gaussian_blur(m, feather)                 # separable, same device
        m = m.clamp(0.0, 1.0)                                    # [N,H,W] pixel mask

        # latent-grid mask for the zeroing — temporal reduction per LTX 8-frame
        # group (temporal_mode; a moving mask stays crisp vs trilinear blur)
        ml = ltx_mask_to_latent(m, T, H, W, mode=temporal_mode).to(
            device=v.device, dtype=v.dtype)

        v_out = v * (1.0 - ml)                                   # zero the masked region
        kept = float((ml < 0.5).float().mean())
        print(f"[LTXInpaintLatent] zeroed masked region of a {W}x{H} latent "
              f"(~{(1 - kept) * 100:.1f}% regenerate); grow {grow}px, feather {feather}px"
              + (", inverted" if invert else "") + ".")
        return ({"samples": v_out}, m)


class LTXRemovalEncode:
    """
    One-node subject removal prep = the validated chain in a single step, with
    the dialed-in values locked so they can't drift: pixel noise-fill → VAE
    encode → latent zero, all on the SAME tight mask.

    - Removes the subject in PIXELS (gone from the latent everywhere → the VAE
      can't smear it from kept cells back into the hole).
    - Fills the hole with decoded (on-manifold) noise at low scale.
    - Encodes, then ZEROS the exact hole in the latent (temporal_mode over the
      8:1 grid) so the model reads nothing there and regenerates fresh.

    Locked internally: tight mask (grow/feather 0), noise_mode = decoded,
    noise_scale = 0.1, fixed seed. Output: (video latent, tight mask). Concat
    the latent with audio → sampler `latents`; grow+blur the mask (KJ
    GrowMaskWithBlur, expand ≥ blur_radius) → optional_denoise_mask, and set the
    sampler's denoise_mask_temporal_mode to match temporal_mode here.
    """

    _SEED = 0
    _NOISE_SCALE = 0.1

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "mask": ("MASK",),
                "vae": ("VAE",),
            },
            "optional": {
                "temporal_mode": (["max", "last", "mean_threshold", "min"], {
                    "default": "last",
                    "tooltip": "Per-frame mask reduction onto the 8:1 latent grid. "
                               "last = causal-aligned crisp (best across cuts); max = "
                               "union (safe, blobs at cuts). Match the sampler's "
                               "denoise_mask_temporal_mode.",
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "MASK")
    RETURN_NAMES = ("latent", "mask")
    FUNCTION     = "prep"
    CATEGORY     = "LTXAVTools/utils"
    DESCRIPTION  = (
        "Subject removal prep in one node: tight pixel noise-fill + VAE encode "
        "+ latent zero on one mask, locked recipe. Latent → concat audio → "
        "sampler; mask → grow+blur → optional_denoise_mask."
    )

    def prep(self, images, mask, vae, temporal_mode="last"):
        # tight pixel noise-fill (grow stays on the denoise side)
        filled, m = LTXNoiseFill().fill(
            images, mask, vae, noise_mode="decoded", grow=0,
            seed=self._SEED, noise_scale=self._NOISE_SCALE,
            _label="LTXRemovalEncode")
        # encode the person-free composite
        lat = vae.encode(filled)
        if isinstance(lat, dict):
            lat = lat.get("samples", lat)
        # zero the exact hole (same tight mask, temporal_mode)
        T, H, W = lat.shape[2], lat.shape[3], lat.shape[4]
        ml = ltx_mask_to_latent(m, T, H, W, mode=temporal_mode).to(
            device=lat.device, dtype=lat.dtype)
        out = lat * (1.0 - ml)
        print(f"[LTXRemovalEncode] tight noise-fill + encode + zero "
              f"(temporal={temporal_mode}); latent {lat.shape[4]}x{lat.shape[3]}x{T}.")
        return ({"samples": out}, m)


class LTXAVTimeRangeMask:
    """Build a denoise mask from TIME RANGES — white = regenerate, black = keep.

    Hand-building these is the annoying part of any inpaint-in-time job: you need
    a frame batch of exactly the right length with the right frames white, and it
    has to land somewhere sensible on the 8:1 latent grid. This does the
    arithmetic and reports what the range actually snapped to.

    Spatially uniform (64x64) — this selects WHEN, not WHERE. Combine with a
    spatial mask if you need both. Nothing is lost by the small size: the
    sampler max-pools onto the latent grid anyway.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("INT", {
                    "default": 125, "min": 1, "max": 100000,
                    "tooltip": "Total PIXEL frames in the clip — the same number "
                               "you gave the empty latent (Frame / Scene Length "
                               "Calculator's frame_count)."}),
                "fps": ("FLOAT", {
                    "default": 25.0, "min": 1.0, "max": 200.0, "step": 0.01}),
                "ranges": ("STRING", {
                    "default": "2-4",
                    "tooltip": "Seconds to REGENERATE, e.g. '2-4' or '2-4, 7-9.5'. "
                               "Everything else is kept."}),
            },
            "optional": {
                "feather_frames": ("INT", {
                    "default": 0, "min": 0, "max": 64,
                    "tooltip": "Pixel frames of ramp at each edge. Note the latent "
                               "reduction is per 8-frame group, so a feather much "
                               "under 8 has little effect."}),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Regenerate everything EXCEPT the ranges."}),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "info")
    FUNCTION = "build"
    CATEGORY = "LTXAVTools/utils"
    DESCRIPTION = (
        "Denoise mask from time ranges (white = regenerate). Reports the latent "
        "and audio spans the range actually lands on, so you can see what the "
        "8:1 grid did to it before spending a render."
    )

    def build(self, frames, fps, ranges, feather_frames=0, invert=False):
        label = "LTXAVTimeRangeMask"
        prof = torch.zeros(int(frames), dtype=torch.float32)
        spans = []
        for tok in ranges.replace(";", ",").split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                a, b = (float(x) for x in tok.split("-"))
            except ValueError:
                print(f"[{label}] ignoring unparseable range {tok!r} "
                      f"(expects 'start-end' in seconds)")
                continue
            i0 = max(0, int(round(a * fps)))
            i1 = min(int(frames), int(round(b * fps)))
            if i1 <= i0:
                print(f"[{label}] range {tok!r} is empty after conversion "
                      f"(frames {i0}..{i1}) — skipped.")
                continue
            prof[i0:i1] = 1.0
            if feather_frames > 0:
                f = min(feather_frames, (i1 - i0) // 2)
                if f > 0:
                    ramp = torch.linspace(0, 1, f + 2)[1:-1]
                    if i0 > 0:
                        prof[i0:i0 + f] = ramp
                    if i1 < frames:
                        prof[i1 - f:i1] = ramp.flip(0)
            spans.append((tok, i0, i1))

        if invert:
            prof = 1.0 - prof

        # What the 8:1 grid will actually do with it (mode=max: any touched
        # latent frame goes free), and the audio span that follows via audio_pos.
        T_lat = (int(frames) - 1) // 8 + 1
        lines = []
        for tok, i0, i1 in spans:
            l0 = 0 if i0 == 0 else (i0 - 1) // 8 + 1
            l1 = min(T_lat, (i1 - 1) // 8 + 1)
            lines.append(
                f"  {tok}s -> px [{i0},{i1}) -> latent [{l0},{l1}) -> audio "
                f"[{audio_pos(max(l0,1), fps) if l0 else 0},{audio_pos(l1, fps)})")
        free = float((prof > 0.5).float().mean()) * 100
        info = chr(10).join(
            [f"{int(frames)} px frames @ {fps:g}fps ({frames / fps:.2f}s), "
             f"{T_lat} latents | {free:.1f}% free"
             + (" (inverted)" if invert else "")] + lines)
        print(f"[{label}] {info}")
        return (prof.view(-1, 1, 1).expand(-1, 64, 64).contiguous(), info)


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
    "LTXInpaintLatent":                 LTXInpaintLatent,
    "LTXNoiseFill":                     LTXNoiseFill,
    "LTXRemovalEncode":                 LTXRemovalEncode,
    "LTXAVTimeRangeMask":               LTXAVTimeRangeMask,
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
    "LTXInpaintLatent":                 "LTX Inpaint Latent",
    "LTXNoiseFill":                     "LTX Noise Fill (pixel removal)",
    "LTXRemovalEncode":                 "LTX Removal Encode",
    "LTXAVTimeRangeMask":               "LTX AV Time Range Mask",
}
