import torch
import numpy as np

try:
    from comfy.ldm.lightricks.av_model import LTXAVModel
    from comfy.ldm.modules.attention import optimized_attention
    _HAS_LTXAV = True
except ImportError:
    _HAS_LTXAV = False

try:
    from comfy_extras.nodes_lt import NestedTensor
    _HAS_NESTED = True
except ImportError:
    try:
        from torch.nested import nested_tensor as NestedTensor
        _HAS_NESTED = True
    except ImportError:
        _HAS_NESTED = False

# Audio VAE constants (verified against LTX VAE config)
_AUDIO_LATENTS_PER_SECOND = 25.0  # 16000 / 160 / 4
_VIDEO_TIME_SCALE = 8


def _get_nested_tensor_class():
    """Return the NestedTensor class used by ComfyUI LTX."""
    try:
        from comfy.ldm.lightricks.av_model import LTXAVModel as _av
        import inspect
        src = inspect.getfile(_av)
        import importlib.util
        # Try comfy_extras first
        from comfy_extras.nodes_lt import LTXVConcatAVLatent
        # Find NestedTensor by checking what LTXAVModel uses
    except Exception:
        pass
    try:
        import comfy.ldm.lightricks.av_model as avm
        # NestedTensor is used inside av_model
        nt = getattr(avm, "NestedTensor", None)
        if nt is not None:
            return nt
    except Exception:
        pass
    return None


def _get_audio_latents_per_second(audio_vae):
    """Derive audio latents per second from the audio VAE if possible, else fall back to 25."""
    try:
        sr = audio_vae.autoencoder.sampling_rate
        hop = audio_vae.autoencoder.mel_hop_length
        downsample = 4
        rate = sr / hop / downsample
        print(f"[LTXAVExtendMask] audio_latents_per_second from VAE: {rate}")
        return rate
    except Exception:
        print(f"[LTXAVExtendMask] could not read audio VAE config, using {_AUDIO_LATENTS_PER_SECOND}")
        return _AUDIO_LATENTS_PER_SECOND


def _build_slope_mask(frame_count, index_start, index_end, slope_len):
    """Build a 1D mask with linear ramps at start and end of generation region."""
    coeffs = [0.0] * frame_count
    index_start = max(0, min(frame_count - 1, index_start))
    index_end = max(index_start, min(frame_count - 1, index_end))
    slope_len = max(1, slope_len)

    # Ramp up
    ramp_start = max(0, index_start - slope_len)
    for i in range(ramp_start, index_start):
        coeffs[i] = (i - ramp_start + 1) / slope_len

    # Plateau
    for i in range(index_start, index_end + 1):
        coeffs[i] = 1.0

    # Ramp down
    ramp_end = min(frame_count, index_end + slope_len + 1)
    for i in range(index_end + 1, ramp_end):
        coeffs[i] = max(0.0, 1.0 - ((i - (index_end + 1) + 1) / slope_len))

    return coeffs


class LTXAVExtendMask:
    """
    Combined AV mask node for LTX2 sliding window generation.

    Extends the LTX SetAudioVideoMaskByTime node with:
    - max_length=pad support (extends latent beyond end_time with zero padding)
    - slope_len ramp applied at both start boundary and pad boundary
    - audio_latents_per_second derived from audio_vae at runtime
    - output_duration_seconds output for downstream math
    - Operates on separated video + audio latents (not NestedTensor) for
      compatibility with KJNodes-style workflows
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_latent": ("LATENT",),
                "audio_latent": ("LATENT",),
                "audio_vae": ("VAE",),
                "video_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2000.0, "step": 0.01,
                                         "tooltip": "Start of generation region in seconds."}),
                "end_time": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 2000.0, "step": 0.01,
                                       "tooltip": "End of generation region in seconds."}),
                "pad_to_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2000.0, "step": 0.01,
                                          "tooltip": "Extend latent with zero padding to this duration. Set to 0 to disable padding."}),
                "slope_len": ("INT", {"default": 3, "min": 1, "max": 100, "step": 1,
                                      "tooltip": "Number of latent frames for ramp transition at mask boundaries."}),
                "strip_input_mask": ("BOOLEAN", {"default": True,
                                                  "tooltip": "Strip existing noise_mask from input latents before applying new mask. Prevents mask accumulation across clips."}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("video_latent", "audio_latent", "output_video_seconds", "audio_latents_per_second")
    FUNCTION = "run"
    CATEGORY = "LTXAVTools"

    def run(self, video_latent, audio_latent, audio_vae, video_fps,
            start_time, end_time, pad_to_time, slope_len, strip_input_mask):

        audio_latents_per_second = _get_audio_latents_per_second(audio_vae)
        video_latents_per_second = video_fps / _VIDEO_TIME_SCALE

        # --- Strip existing masks if requested ---
        v = video_latent.copy()
        a = audio_latent.copy()
        if strip_input_mask:
            v.pop("noise_mask", None)
            a.pop("noise_mask", None)

        video_samples = v["samples"]
        audio_samples = a["samples"]

        if video_samples.ndim != 5:
            raise ValueError(f"Expected 5D video latent [B,C,T,H,W], got {video_samples.shape}")
        if audio_samples.ndim != 4:
            raise ValueError(f"Expected 4D audio latent [B,C,T,F], got {audio_samples.shape}")

        B, C_v, T_v, H, W = video_samples.shape
        B, C_a, T_a, F_a = audio_samples.shape

        # --- Compute frame indices ---
        video_pixel_frame_count = (T_v - 1) * _VIDEO_TIME_SCALE + 1
        xp = np.array([0] + list(range(1, video_pixel_frame_count + _VIDEO_TIME_SCALE, _VIDEO_TIME_SCALE)))

        vid_start_px = int(round(start_time * video_fps))
        vid_end_px = int(round(end_time * video_fps))
        vid_start_idx = int(np.searchsorted(xp, vid_start_px, side="left"))
        vid_end_idx = int(np.searchsorted(xp, vid_end_px, side="right")) - 1

        aud_start_idx = int(round(start_time * audio_latents_per_second))
        aud_end_idx = int(round(end_time * audio_latents_per_second))

        # --- Pad latents if pad_to_time > current duration ---
        if pad_to_time > 0:
            # Video padding
            required_vid_frames = int(round((pad_to_time * video_fps - 1) / _VIDEO_TIME_SCALE)) + 1
            if required_vid_frames > T_v:
                pad_v = required_vid_frames - T_v
                video_samples = torch.cat([
                    video_samples,
                    torch.zeros(B, C_v, pad_v, H, W, dtype=video_samples.dtype, device=video_samples.device)
                ], dim=2)
                T_v = video_samples.shape[2]

            # Audio padding
            required_aud_frames = int(round(pad_to_time * audio_latents_per_second)) + 1
            if required_aud_frames > T_a:
                pad_a = required_aud_frames - T_a
                audio_samples = torch.cat([
                    audio_samples,
                    torch.zeros(B, C_a, pad_a, F_a, dtype=audio_samples.dtype, device=audio_samples.device)
                ], dim=2)
                T_a = audio_samples.shape[2]

        # Clamp indices to actual sizes
        vid_start_idx = max(0, min(vid_start_idx, T_v - 1))
        vid_end_idx = max(vid_start_idx, min(vid_end_idx, T_v - 1))
        aud_start_idx = max(0, min(aud_start_idx, T_a - 1))
        aud_end_idx = max(aud_start_idx, min(aud_end_idx, T_a - 1))

        print(
            f"[LTXAVExtendMask] video T={T_v} mask=[{vid_start_idx},{vid_end_idx}] | "
            f"audio T={T_a} mask=[{aud_start_idx},{aud_end_idx}]"
        )

        # --- Build video mask with slope ---
        vid_mask_coeffs = _build_slope_mask(T_v, vid_start_idx, vid_end_idx, slope_len)
        video_mask = torch.zeros(B, 1, T_v, 1, 1, dtype=video_samples.dtype, device=video_samples.device)
        for i, c in enumerate(vid_mask_coeffs):
            video_mask[:, :, i, :, :] = c
        # Ensure padded region is fully masked
        if pad_to_time > 0:
            orig_T_v = int(round((end_time * video_fps - 1) / _VIDEO_TIME_SCALE)) + 1
            if orig_T_v < T_v:
                video_mask[:, :, orig_T_v:, :, :] = 1.0

        # --- Build audio mask with slope ---
        aud_mask_coeffs = _build_slope_mask(T_a, aud_start_idx, aud_end_idx, slope_len)
        audio_mask = torch.zeros(B, C_a, T_a, F_a, dtype=audio_samples.dtype, device=audio_samples.device)
        for i, c in enumerate(aud_mask_coeffs):
            audio_mask[:, :, i, :] = c
        # Ensure padded region is fully masked
        if pad_to_time > 0:
            orig_T_a = int(round(end_time * audio_latents_per_second)) + 1
            if orig_T_a < T_a:
                audio_mask[:, :, orig_T_a:, :] = 1.0

        v["samples"] = video_samples
        v["noise_mask"] = video_mask
        a["samples"] = audio_samples
        a["noise_mask"] = audio_mask

        output_seconds = T_v_to_seconds(T_v, video_fps)

        return (v, a, output_seconds, audio_latents_per_second)


def T_v_to_seconds(T_v, video_fps):
    pixel_frames = (T_v - 1) * _VIDEO_TIME_SCALE + 1
    return pixel_frames / video_fps


NODE_CLASS_MAPPINGS = {
    "LTXAVExtendMask": LTXAVExtendMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXAVExtendMask": "LTXAV Extend Mask",
}
