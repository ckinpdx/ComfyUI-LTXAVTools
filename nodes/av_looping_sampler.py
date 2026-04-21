"""
LTXVAVLoopingSampler — AV-aware looping sampler for long-form I2V generation.

Generates video + audio jointly (cross-modal) using NestedTensor AV latents.
Temporal tiling with audio overlap carry-over maintains continuity across chunks.

Audio alignment notes:
  - AUDIO_LATENTS_PER_SECOND = 25.0 (fixed for LTX AV)
  - First video latent = 1 pixel frame; subsequent = 8 pixel frames (LTX asymmetry)
  - Each extend chunk generates 7 fewer audio frames than the global timeline expects
    due to the first-frame treatment. Fixed by: drop 1 audio frame (frame 0) + pad 7.
  - Audio blend overlap = a_overlap + 6, giving exact num_new_v * 8 frame growth per chunk.

Spatial tiling with AV: audio accumulated from tile (0,0) only.
"""

import copy
import math

import comfy
import comfy.utils
import node_helpers
import torch
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced, SplitSigmas
from comfy_extras.nodes_lt import (
    EmptyLTXVLatentVideo,
    LTXVAddGuide,
    LTXVCropGuides,
    get_noise_mask,
)

AUDIO_LATENTS_PER_SECOND = 25.0


# ---------------------------------------------------------------------------
# Audio / latent utilities
# ---------------------------------------------------------------------------

def _audio_frames_for_video_chunk(v_latent_frames, fps):
    """
    Audio latent frames for a video chunk of v_latent_frames, accounting for
    the first-frame asymmetry (pos 0 = 1 px, all others = 8 px).
    """
    px = (v_latent_frames - 1) * 8 + 1
    return max(1, round(px / fps * AUDIO_LATENTS_PER_SECOND))


def _audio_overlap_frames(video_overlap_latents, fps):
    """
    Audio latent frames corresponding to the video overlap region as it
    appears in an extend chunk's LOCAL frame (pos 0 = 1 px, rest = 8 px each).
    """
    px = 1 + (video_overlap_latents - 1) * 8 if video_overlap_latents > 1 else 1
    return max(1, round(px / fps * AUDIO_LATENTS_PER_SECOND))


def _select_video_frames(latent_dict, start, end):
    """Slice video latent frames [start, end] inclusive. Supports negative indices."""
    s = latent_dict.copy()
    v = s["samples"]
    T = v.shape[2]
    si = T + start if start < 0 else start
    ei = T + end   if end   < 0 else end
    si = max(0, min(si, T - 1))
    ei = max(0, min(ei, T - 1))
    s["samples"] = v[:, :, si:ei + 1]
    if "noise_mask" in s and s["noise_mask"] is not None:
        s["noise_mask"] = s["noise_mask"][:, :, si:ei + 1]
    return s


def _linear_overlap_blend(t1, t2, overlap, axis=2):
    """
    Linear crossfade between t1 and t2 over `overlap` elements along `axis`.
    Result length = len(t1) + len(t2) - overlap.
    """
    if overlap <= 0:
        return torch.cat([t1, t2], dim=axis)

    alpha = torch.linspace(1, 0, overlap + 2, device=t1.device, dtype=t1.dtype)[1:-1]
    shape = [1] * t1.dim()
    shape[axis] = overlap
    alpha = alpha.reshape(shape)

    sl = [slice(None)] * t1.dim()

    sl_keep1  = sl.copy(); sl_keep1[axis]  = slice(None, -overlap)
    sl_ovlp1  = sl.copy(); sl_ovlp1[axis]  = slice(-overlap, None)
    sl_ovlp2  = sl.copy(); sl_ovlp2[axis]  = slice(None, overlap)
    sl_rest2  = sl.copy(); sl_rest2[axis]  = slice(overlap, None)

    blended = alpha * t1[tuple(sl_ovlp1)] + (1 - alpha) * t2[tuple(sl_ovlp2)]
    return torch.cat([t1[tuple(sl_keep1)], blended, t2[tuple(sl_rest2)]], dim=axis)


def _get_raw_conds(guider):
    if hasattr(guider, "raw_conds"):
        return guider.raw_conds
    raw_pos = guider.original_conds["positive"]
    raw_neg = guider.original_conds["negative"]
    positive = [[raw_pos[0]["cross_attn"], copy.deepcopy(raw_pos[0])]]
    negative = [[raw_neg[0]["cross_attn"], copy.deepcopy(raw_neg[0])]]
    guider.raw_conds = (positive, negative)
    return positive, negative


# ---------------------------------------------------------------------------
# LTXVAVLoopingSampler
# ---------------------------------------------------------------------------

class LTXVAVLoopingSampler:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model":      ("MODEL",),
                "vae":        ("VAE",),
                "noise":      ("NOISE",),
                "sampler":    ("SAMPLER",),
                "sigmas":     ("SIGMAS",),
                "guider":     ("GUIDER",),
                "latents":    ("LATENT",),
                "temporal_tile_size": ("INT", {
                    "default": 80, "min": 24, "max": 1000, "step": 8,
                    "tooltip": "Pixel frames per temporal tile (excluding overlap).",
                }),
                "temporal_overlap": ("INT", {
                    "default": 24, "min": 16, "max": 80, "step": 8,
                    "tooltip": "Pixel frames of overlap between temporal tiles.",
                }),
                "guiding_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "temporal_overlap_cond_strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Noise mask strength for video overlap carry-over region.",
                }),
                "audio_overlap_cond_strength": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Noise mask strength for audio overlap carry-over. "
                               "Higher values freeze the carry-over region more strongly. "
                               "Try 0.9-1.0 if chunk boundaries sound rough.",
                }),
                "cond_image_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "horizontal_tiles": ("INT", {"default": 1, "min": 1, "max": 6}),
                "vertical_tiles":   ("INT", {"default": 1, "min": 1, "max": 6}),
                "spatial_overlap":  ("INT", {"default": 1, "min": 1, "max": 8}),
                "video_fps": ("FLOAT", {
                    "default": 25.0, "min": 1.0, "max": 120.0, "step": 0.01,
                    "tooltip": "Must match the fps of the AV latent. Used for audio frame alignment.",
                }),
            },
            "optional": {
                "optional_cond_images":            ("IMAGE",),
                "optional_guiding_latents":        ("LATENT",),
                "optional_negative_index_latents": ("LATENT",),
                "optional_positive_conditionings": ("CONDITIONING",),
                "optional_normalizing_latents":    ("LATENT",),
                "adain_factor": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "guiding_start_step": ("INT", {"default": 0,    "min": 0, "max": 1000}),
                "guiding_end_step":   ("INT", {"default": 1000, "min": 0, "max": 1000}),
                "optional_cond_image_indices": ("STRING", {"default": "0"}),
                "audio_cond_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Strength of conditioning on the input audio latent. "
                               "0.0 = ignore input audio (generate freely). "
                               "1.0 = fully condition on input audio (no regeneration). "
                               "Only active when the input AV latent contains non-zero audio.",
                }),
            },
        }

    RETURN_TYPES  = ("LATENT",)
    RETURN_NAMES  = ("denoised_output",)
    FUNCTION      = "sample"
    CATEGORY      = "LTXAVTools/sampling"

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get_ltxav_model(self, model):
        dm = model.model.diffusion_model
        if dm.__class__.__name__ != "LTXAVModel":
            raise ValueError(
                f"[LTXVAVLoopingSampler] Expected LTXAVModel, got {dm.__class__.__name__}."
            )
        return dm

    def _split_av(self, nested, ltxav):
        return ltxav.separate_audio_and_video_latents(nested.tensors, None)

    def _combine_av(self, video, audio, ltxav):
        return NestedTensor(ltxav.recombine_audio_and_video_latents(video, audio))

    def _add_latent_guide(self, vae, positive, negative, latent_dict,
                          guide_latent_dict, latent_idx, strength):
        """
        Add a pre-encoded video latent as a guide via LTXVAddGuide.append_keyframe.
        Does not add IC-LoRA attention entries — sufficient for overlap and
        negative-index conditioning. For full IC-LoRA guided generation use
        LTXVAddLatentGuide from ComfyUI-LTXVideo instead.
        """
        guide   = guide_latent_dict["samples"]
        ts      = vae.downscale_index_formula
        time_sc = ts[0]

        if latent_idx <= 0:
            frame_idx = latent_idx * time_sc
        else:
            frame_idx = 1 + (latent_idx - 1) * time_sc

        noise_mask    = get_noise_mask(latent_dict)
        latent_tensor = latent_dict["samples"]
        guide_mask    = guide_latent_dict.get("noise_mask", None)

        positive, negative, latent_tensor, noise_mask = LTXVAddGuide.append_keyframe(
            positive=positive, negative=negative,
            frame_idx=frame_idx,
            latent_image=latent_tensor,
            noise_mask=noise_mask,
            guiding_latent=guide,
            strength=strength,
            scale_factors=ts,
            guide_mask=guide_mask,
        )
        return positive, negative, {"samples": latent_tensor, "noise_mask": noise_mask}

    def _build_av_noise_mask(self, video_latent_dict, audio_mask_tensor, ltxav):
        """Combine video noise mask + audio mask into a NestedTensor noise mask."""
        vm = get_noise_mask(video_latent_dict)
        return NestedTensor(ltxav.recombine_audio_and_video_latents(vm, audio_mask_tensor))

    def _run_sampling(self, noise, guider, sampler, sigmas,
                      av_init, guiding_start_step, guiding_end_step):
        high, rest = SplitSigmas().get_sigmas(sigmas, guiding_start_step)
        mid,  low  = SplitSigmas().get_sigmas(rest, guiding_end_step - guiding_start_step)

        current = av_init
        if len(high) > 1:
            _, current = SamplerCustomAdvanced().sample(noise, guider, sampler, high, current)
        if len(mid) > 1:
            _, current = SamplerCustomAdvanced().sample(noise, guider, sampler, mid,  current)
        return current

    def _crop_and_split(self, av_latent_dict, positive, negative, ltxav):
        """
        Separate video and audio, crop video guide tokens, return both.
        Audio has no guide tokens appended so passes through unchanged.
        """
        samples = av_latent_dict["samples"]
        if isinstance(samples, NestedTensor):
            video_raw, audio_out = self._split_av(samples, ltxav)
        else:
            video_raw = samples
            audio_out = None

        positive, negative, video_cropped = LTXVCropGuides.execute(
            positive, negative, {"samples": video_raw}
        )
        return video_cropped["samples"], audio_out, positive, negative

    def _debug_chunk(self, chunk_idx, v_start, v_end, T_v, a_start, a_end, T_a, fps):
        px       = (T_v - 1) * 8 + 1
        expected = round(px / fps * AUDIO_LATENTS_PER_SECOND)
        delta    = T_a - expected
        status   = "OK" if delta == 0 else f"MISMATCH delta={delta:+d}"
        print(
            f"[LTXVAVLoopingSampler] chunk={chunk_idx} "
            f"v=[{v_start},{v_end}] ({T_v} latents) | "
            f"a=[{a_start},{a_end}] ({T_a} latents) | "
            f"expected_a={expected} | {status}"
        )

    def _calculate_tile_seed(self, first_seed, start_index,
                             vert, horiz, v, h, offset):
        return first_seed + start_index * (vert * horiz) + v * horiz + h + offset

    def _get_per_tile_value(self, lst, idx):
        return lst[min(idx, len(lst) - 1)]

    def _parse_ints(self, s, default="0", total_size=None):
        if not s:
            s = default
        vals = [int(x.strip()) for x in s.split(",")]
        if total_size is not None:
            vals = [v + total_size if v < 0 else v for v in vals]
        return vals

    def _prepare_guider(self, guider, optional_positive_conditionings, chunk_index):
        if optional_positive_conditionings is None:
            return guider
        new_g = copy.copy(guider)
        positive, negative = _get_raw_conds(guider)
        idx = min(chunk_index, len(optional_positive_conditionings) - 1)
        new_g.set_conds(optional_positive_conditionings[idx], negative)
        new_g.raw_conds = (optional_positive_conditionings[idx], negative)
        return new_g

    def _calculate_keyframe_per_tile_indices(self, keyframe_indices,
                                              temporal_tile_size, temporal_overlap,
                                              num_frames):
        result = []
        for kf in keyframe_indices:
            if kf >= num_frames:
                continue
            if kf < temporal_tile_size - 7:
                result.append((0, kf))
                continue
            tile_step  = temporal_tile_size - temporal_overlap
            tile_index = 1
            while True:
                tile_start = tile_index * tile_step - 7
                tile_end   = temporal_tile_size + tile_index * tile_step - 1 - 7
                if kf <= tile_end:
                    in_tile = kf - tile_start - 7
                    if in_tile < temporal_overlap:
                        tile_index -= 1
                        if tile_index == 0:
                            in_tile = kf
                        else:
                            in_tile = kf - (tile_start - tile_step) - 7
                    result.append((tile_index, in_tile))
                    break
                tile_index += 1
        return result

    # ------------------------------------------------------------------ #
    # First chunk (I2V)                                                    #
    # ------------------------------------------------------------------ #

    def _sample_first_chunk(
        self, model, vae, noise, sampler, sigmas, guider,
        video_tile_latent, audio_full, ltxav,
        time_scale, width_scale, height_scale,
        tile_h, tile_w, temporal_tile_size, fps,
        cond_images, cond_image_strength,
        neg_idx_latents, neg_idx, neg_idx_strength,
        guiding_start_step, guiding_end_step,
        guiding_latents, guiding_strength,
        keyframes, kf_idx_str,
        audio_cond_strength=0.0,
    ):
        guider = copy.copy(guider)
        guider.original_conds = copy.deepcopy(guider.original_conds)
        positive, negative = _get_raw_conds(guider)

        T_v = min(temporal_tile_size, video_tile_latent["samples"].shape[2])
        num_px = (T_v - 1) * time_scale + 1

        # video init — use initialization latent if non-empty, else empty
        v_samples = video_tile_latent["samples"]
        if v_samples.shape[2] >= T_v and v_samples.abs().sum() > 0:
            video_init = {"samples": v_samples[:, :, :T_v].clone()}
        else:
            video_init = EmptyLTXVLatentVideo.execute(
                width=tile_w * width_scale, height=tile_h * height_scale,
                length=num_px, batch_size=1,
            )[0]

        # I2V: encode image at frame 0
        if cond_images is not None and kf_idx_str:
            kf_indices = [int(x) for x in kf_idx_str.split(",") if x.strip()]
            for img, idx in zip(cond_images, kf_indices):
                if idx == 0:
                    encoded = vae.encode(img.unsqueeze(0)[:, :, :, :3])
                    video_init["samples"][:, :, :encoded.shape[2]] = encoded
                    mask = video_init.get("noise_mask") or torch.ones(
                        (1, 1, T_v, 1, 1), device=video_init["samples"].device
                    )
                    mask[:, :, :encoded.shape[2]] = 1.0 - cond_image_strength
                    video_init["noise_mask"] = mask
                else:
                    positive, negative, video_init = LTXVAddGuide.execute(
                        positive, negative, vae, video_init,
                        img.unsqueeze(0), idx, cond_image_strength,
                    )

        # guiding latents (IC-LoRA) — first chunk starts at latent_idx 0
        if guiding_latents is not None:
            g_chunk = _select_video_frames(
                guiding_latents, 0,
                min(T_v - 1, guiding_latents["samples"].shape[2] - 1),
            )
            positive, negative, video_init = self._add_latent_guide(
                vae, positive, negative, video_init, g_chunk, 0, guiding_strength
            )

        # negative index
        if neg_idx_latents is not None:
            positive, negative, video_init = self._add_latent_guide(
                vae, positive, negative, video_init,
                neg_idx_latents, neg_idx, neg_idx_strength,
            )

        # audio init
        B   = video_init["samples"].shape[0]
        C_a = audio_full.shape[1]
        F_s = audio_full.shape[3]
        T_a = _audio_frames_for_video_chunk(T_v, fps)
        if audio_cond_strength > 0.0:
            T_a = min(T_a, audio_full.shape[2])
            audio_init = audio_full[:, :, :T_a, :].clone()
        else:
            audio_init = torch.zeros(B, C_a, T_a, F_s,
                                     device=audio_full.device, dtype=audio_full.dtype)

        self._debug_chunk(0, 0, T_v - 1, T_v, 0, T_a - 1, T_a, fps)

        # combine and sample
        audio_mask_val = 1.0 - audio_cond_strength
        audio_mask = torch.full((B, 1, T_a, F_s), audio_mask_val,
                                device=audio_full.device, dtype=audio_full.dtype)

        video_has_mask = "noise_mask" in video_init and video_init["noise_mask"] is not None
        av_init = {"samples": self._combine_av(video_init["samples"], audio_init, ltxav)}
        if video_has_mask or audio_cond_strength > 0.0:
            if not video_has_mask:
                # need a paired video mask (all-ones = full denoising for video)
                video_init["noise_mask"] = torch.ones(
                    B, 1, T_v, 1, 1,
                    device=video_init["samples"].device, dtype=video_init["samples"].dtype,
                )
            av_init["noise_mask"] = self._build_av_noise_mask(video_init, audio_mask, ltxav)

        guider.set_conds(positive, negative)
        av_out = self._run_sampling(noise, guider, sampler, sigmas, av_init,
                                    guiding_start_step, guiding_end_step)

        video_out, audio_out, pos_out, neg_out = self._crop_and_split(
            av_out, positive, negative, ltxav
        )
        return {"samples": video_out}, audio_out

    # ------------------------------------------------------------------ #
    # Extend chunks                                                        #
    # ------------------------------------------------------------------ #

    def _sample_extend_chunk(
        self, model, vae, noise, sampler, sigmas, guider,
        video_acc, audio_acc,
        video_tile_latent, audio_full, ltxav,
        v_start, v_end,
        time_scale, width_scale, height_scale,
        tile_h, tile_w,
        temporal_tile_size, temporal_overlap, fps,
        overlap_cond_strength,
        audio_overlap_cond_strength,
        neg_idx_latents, neg_idx, neg_idx_strength,
        guiding_start_step, guiding_end_step,
        guiding_latents, guiding_strength,
        keyframes, kf_idx_str, cond_image_strength,
        chunk_index,
        audio_cond_strength=0.0,
    ):
        guider = copy.copy(guider)
        guider.original_conds = copy.deepcopy(guider.original_conds)
        positive, negative = _get_raw_conds(guider)

        num_new_v = min(v_end, video_tile_latent["samples"].shape[2]) - v_start
        T_v_chunk = temporal_overlap + num_new_v

        # video: last overlap frames from accumulated output as guide
        overlap_video = _select_video_frames(video_acc, -temporal_overlap, -1)
        num_px_total  = (T_v_chunk - 1) * time_scale + 1

        video_init = EmptyLTXVLatentVideo.execute(
            width=tile_w * width_scale, height=tile_h * height_scale,
            length=num_px_total, batch_size=1,
        )[0]

        positive, negative, video_init = self._add_latent_guide(
            vae, positive, negative, video_init,
            overlap_video, latent_idx=0, strength=overlap_cond_strength,
        )

        if guiding_latents is not None:
            g_chunk = _select_video_frames(
                guiding_latents, v_start,
                min(v_end - 1, guiding_latents["samples"].shape[2] - 1),
            )
            positive, negative, video_init = self._add_latent_guide(
                vae, positive, negative, video_init, g_chunk,
                latent_idx=overlap_video["samples"].shape[2],
                strength=guiding_strength,
            )

        if keyframes is not None and kf_idx_str:
            kf_indices = [int(x) for x in kf_idx_str.split(",") if x.strip()]
            for img, idx in zip(keyframes, kf_indices):
                positive, negative, video_init = LTXVAddGuide.execute(
                    positive, negative, vae, video_init,
                    img.unsqueeze(0), idx, cond_image_strength,
                )

        if neg_idx_latents is not None:
            positive, negative, video_init = self._add_latent_guide(
                vae, positive, negative, video_init,
                neg_idx_latents, neg_idx, neg_idx_strength,
            )

        # audio init
        B   = video_init["samples"].shape[0]
        C_a = audio_acc.shape[1]
        F_s = audio_acc.shape[3]
        dev, dty = audio_acc.device, audio_acc.dtype

        a_overlap = _audio_overlap_frames(temporal_overlap, fps)
        a_overlap = min(a_overlap, audio_acc.shape[2])
        T_a_chunk = _audio_frames_for_video_chunk(T_v_chunk, fps)
        T_a_new   = max(1, T_a_chunk - a_overlap)

        audio_carry = audio_acc[:, :, -a_overlap:, :].clone()

        # new-region audio: conditioned on input if audio_cond_strength > 0
        a_global_new_start = audio_acc.shape[2]
        a_global_new_end   = a_global_new_start + T_a_new
        audio_new = torch.zeros(B, C_a, T_a_new, F_s, device=dev, dtype=dty)
        if audio_cond_strength > 0.0 and a_global_new_start < audio_full.shape[2]:
            a_end_clamped = min(a_global_new_end, audio_full.shape[2])
            available = a_end_clamped - a_global_new_start
            audio_new[:, :, :available, :] = audio_full[
                :, :, a_global_new_start:a_end_clamped, :
            ]

        audio_init  = torch.cat([audio_carry, audio_new], dim=2)

        # noise mask: low on carry-over, audio_cond_strength-derived on new
        audio_mask = torch.ones(B, 1, T_a_chunk, F_s, device=dev, dtype=dty)
        audio_mask[:, :, :a_overlap, :] = 1.0 - audio_overlap_cond_strength
        audio_mask[:, :, a_overlap:,  :] = 1.0 - audio_cond_strength

        # ref_audio: inject carry-over as guide tokens so the model attends to
        # the spectral character of the previous chunk (audio equivalent of video
        # guide tokens). Complements the noise mask with identity-level continuity.
        ref_tokens = audio_carry.permute(0, 2, 1, 3).reshape(B, a_overlap, C_a * F_s)
        ref_audio  = {"tokens": ref_tokens}
        positive   = node_helpers.conditioning_set_values(positive, {"ref_audio": ref_audio})
        negative   = node_helpers.conditioning_set_values(negative, {"ref_audio": ref_audio})

        a_start_global = audio_acc.shape[2] - a_overlap
        self._debug_chunk(
            chunk_index, v_start, v_end - 1, T_v_chunk,
            a_start_global, a_start_global + T_a_chunk - 1, T_a_chunk, fps,
        )

        av_init = {"samples": self._combine_av(video_init["samples"], audio_init, ltxav)}
        av_init["noise_mask"] = self._build_av_noise_mask(video_init, audio_mask, ltxav)

        guider.set_conds(positive, negative)
        av_out = self._run_sampling(noise, guider, sampler, sigmas, av_init,
                                    guiding_start_step, guiding_end_step)

        video_out, audio_out, _, _ = self._crop_and_split(av_out, positive, negative, ltxav)

        # --- stitch video ---
        # drop local frame 0 (1 px stub from first-frame treatment)
        video_trimmed = video_out[:, :, 1:]
        # linear blend: video overlap - 1 frames
        video_blend_overlap = temporal_overlap - 1
        video_result = _linear_overlap_blend(
            video_acc["samples"], video_trimmed, video_blend_overlap, axis=2
        )

        # --- stitch audio ---
        # Drop frame 0 (first-frame asymmetry stub), pad 1 to restore exact
        # T_a_chunk length (drop-1 leaves T_a_chunk-1; we need T_a_chunk).
        audio_trimmed = audio_out[:, :, 1:, :]
        pad1 = audio_trimmed[:, :, -1:, :]
        audio_body = torch.cat([audio_trimmed, pad1], dim=2)  # T_a_chunk frames

        # Use model's regenerated carry-over as the bridge rather than hard-joining
        # audio_acc's tail to the new frames. The model generated audio_body as one
        # coherent sequence (carry-over + new), so replacing acc's tail with it
        # avoids a latent-space discontinuity. With high audio_overlap_cond_strength
        # the regenerated carry-over is nearly identical to the original.
        # Result length = (acc - a_overlap) + T_a_chunk = acc + num_new_v*8 exactly.
        audio_head   = audio_acc[:, :, :-a_overlap, :]
        audio_result = torch.cat([audio_head, audio_body], dim=2)

        return {"samples": video_result}, audio_result

    # ------------------------------------------------------------------ #
    # Temporal loop                                                        #
    # ------------------------------------------------------------------ #

    def _process_temporal_chunks(
        self,
        model, vae, noise, sampler, sigmas, guider,
        video_tile_latent, audio_full, ltxav,
        time_scale, width_scale, height_scale,
        tile_h, tile_w,
        temporal_tile_size, temporal_overlap, fps,
        overlap_cond_strength,
        audio_overlap_cond_strength,
        cond_images, cond_image_strength,
        neg_idx_latents, neg_idx, neg_idx_strength,
        optional_positive_conditionings,
        guiding_start_step, guiding_end_step,
        guiding_latents, guiding_strength,
        keyframe_per_tile_indices, optional_keyframes,
        first_seed, v, h, vert_tiles, horiz_tiles,
        per_tile_seed_offsets,
        audio_cond_strength=0.0,
    ):
        T_v  = video_tile_latent["samples"].shape[2]
        step = temporal_tile_size - temporal_overlap
        video_acc = None
        audio_acc = None
        chunk_idx = 0

        starts = range(0, T_v + step, step)
        ends   = range(temporal_tile_size, T_v + temporal_tile_size, step)

        for i_t, (v_start, v_end) in enumerate(zip(starts, ends)):
            v_end = min(v_end, T_v)
            if v_start >= T_v:
                break

            seed_off  = self._get_per_tile_value(per_tile_seed_offsets, i_t)
            noise.seed = self._calculate_tile_seed(
                first_seed, v_start, vert_tiles, horiz_tiles, v, h, seed_off
            )
            cur_guider = self._prepare_guider(guider, optional_positive_conditionings, chunk_idx)

            # keyframes for this chunk
            chunk_kf_in_tile = [
                in_idx for (t_idx, in_idx) in keyframe_per_tile_indices if t_idx == i_t
            ]
            if chunk_kf_in_tile and optional_keyframes is not None:
                chunk_kf_images = torch.cat([
                    optional_keyframes[ki].unsqueeze(0)
                    for ki, (t_idx, _) in enumerate(keyframe_per_tile_indices)
                    if t_idx == i_t
                ])
                kf_str = ",".join(str(x) for x in chunk_kf_in_tile)
            else:
                chunk_kf_images = None
                kf_str = ""

            print(f"[LTXVAVLoopingSampler] temporal chunk {i_t} v=[{v_start},{v_end - 1}]")

            shared = dict(
                model=model, vae=vae, noise=noise, sampler=sampler,
                sigmas=sigmas, guider=cur_guider,
                video_tile_latent=video_tile_latent, audio_full=audio_full, ltxav=ltxav,
                time_scale=time_scale, width_scale=width_scale, height_scale=height_scale,
                tile_h=tile_h, tile_w=tile_w, fps=fps,
                neg_idx_latents=neg_idx_latents, neg_idx=neg_idx, neg_idx_strength=neg_idx_strength,
                guiding_start_step=guiding_start_step, guiding_end_step=guiding_end_step,
                guiding_latents=guiding_latents, guiding_strength=guiding_strength,
                keyframes=chunk_kf_images, kf_idx_str=kf_str,
                audio_cond_strength=audio_cond_strength,
            )

            if v_start == 0:
                video_acc, audio_acc = self._sample_first_chunk(
                    **shared,
                    temporal_tile_size=temporal_tile_size,
                    cond_images=cond_images, cond_image_strength=cond_image_strength,
                )
            else:
                video_acc, audio_acc = self._sample_extend_chunk(
                    **shared,
                    video_acc=video_acc, audio_acc=audio_acc,
                    v_start=v_start, v_end=v_end,
                    temporal_tile_size=temporal_tile_size,
                    temporal_overlap=temporal_overlap,
                    overlap_cond_strength=overlap_cond_strength,
                    audio_overlap_cond_strength=audio_overlap_cond_strength,
                    cond_image_strength=cond_image_strength,
                    chunk_index=chunk_idx,
                )
            chunk_idx += 1

        return video_acc, audio_acc

    # ------------------------------------------------------------------ #
    # Entry point                                                          #
    # ------------------------------------------------------------------ #

    def sample(
        self,
        model, vae, noise, sampler, sigmas, guider, latents,
        guiding_strength, adain_factor,
        temporal_tile_size, temporal_overlap, temporal_overlap_cond_strength,
        horizontal_tiles, vertical_tiles, spatial_overlap,
        video_fps=25.0,
        audio_overlap_cond_strength=0.9,
        audio_cond_strength=0.0,
        optional_cond_images=None, cond_image_strength=1.0,
        optional_guiding_latents=None,
        optional_negative_index_latents=None,
        optional_negative_index_strength=1.0,
        optional_positive_conditionings=None,
        optional_normalizing_latents=None,
        guiding_start_step=0, guiding_end_step=1000,
        optional_cond_image_indices="0",
        per_tile_seed_offsets="0",          # hidden
        optional_negative_index_strength_v=1.0,  # alias
    ):
        samples = latents["samples"]
        if not isinstance(samples, NestedTensor):
            raise ValueError(
                "[LTXVAVLoopingSampler] Input latent must be an AV NestedTensor. "
                "Use LTXVLoopingSampler for video-only generation."
            )

        ltxav = self._get_ltxav_model(model)
        video_samples, audio_samples = self._split_av(samples, ltxav)

        B, C, frames, height, width = video_samples.shape
        time_scale, width_scale, height_scale = vae.downscale_index_formula

        tile_size_v = temporal_tile_size // time_scale
        overlap_v   = temporal_overlap   // time_scale
        first_seed  = noise.seed

        per_tile_seed_offsets_list = self._parse_ints(per_tile_seed_offsets, "0")

        kf_indices = self._parse_ints(
            optional_cond_image_indices, "0",
            total_size=frames * time_scale - 7,
        )
        kf_per_tile = self._calculate_keyframe_per_tile_indices(
            kf_indices, temporal_tile_size, temporal_overlap, frames * time_scale - 7
        )

        if optional_cond_images is not None:
            optional_keyframes = (
                comfy.utils.common_upscale(
                    optional_cond_images.movedim(-1, 1),
                    width * width_scale, height * height_scale,
                    "bilinear", crop="center",
                ).movedim(1, -1).clamp(0, 1)
            )
        else:
            optional_keyframes = None

        # spatial tile sizes
        base_tile_h = (height + (vertical_tiles   - 1) * spatial_overlap) // vertical_tiles
        base_tile_w = (width  + (horizontal_tiles - 1) * spatial_overlap) // horizontal_tiles

        video_final   = None
        video_weights = None
        audio_final   = None

        if horizontal_tiles > 1 or vertical_tiles > 1:
            print(
                "[LTXVAVLoopingSampler] Spatial tiling active: "
                "audio accumulated from tile (0,0) only."
            )

        for v in range(vertical_tiles):
            for h in range(horizontal_tiles):
                hs = h * (base_tile_w - spatial_overlap)
                vs = v * (base_tile_h - spatial_overlap)
                he = min(hs + base_tile_w, width)  if h < horizontal_tiles - 1 else width
                ve = min(vs + base_tile_h, height) if v < vertical_tiles   - 1 else height
                th, tw = ve - vs, he - hs

                print(f"[LTXVAVLoopingSampler] spatial tile v={v} h={h} "
                      f"({vs}:{ve}, {hs}:{he})")

                video_tile = {"samples": video_samples[:, :, :, vs:ve, hs:he]}

                tile_guide = None
                if optional_guiding_latents is not None:
                    tile_guide = {
                        "samples": optional_guiding_latents["samples"][:, :, :, vs:ve, hs:he]
                    }

                tile_neg_idx = None
                neg_idx = -1
                if optional_negative_index_latents is not None:
                    n = optional_negative_index_latents["samples"]
                    tile_neg_idx = {"samples": n[:, :, :, vs:ve, hs:he]}
                    neg_idx = -tile_neg_idx["samples"].shape[2]

                tile_kf = None
                if optional_keyframes is not None:
                    tile_kf = optional_keyframes[
                        :,
                        vs * height_scale:ve * height_scale,
                        hs * width_scale :he * width_scale,
                        :,
                    ]

                tile_video_out, tile_audio_out = self._process_temporal_chunks(
                    model=model, vae=vae, noise=noise,
                    sampler=sampler, sigmas=sigmas, guider=guider,
                    video_tile_latent=video_tile,
                    audio_full=audio_samples, ltxav=ltxav,
                    time_scale=time_scale, width_scale=width_scale, height_scale=height_scale,
                    tile_h=th, tile_w=tw,
                    temporal_tile_size=tile_size_v,
                    temporal_overlap=overlap_v,
                    fps=video_fps,
                    overlap_cond_strength=temporal_overlap_cond_strength,
                    audio_overlap_cond_strength=audio_overlap_cond_strength,
                    audio_cond_strength=audio_cond_strength,
                    cond_images=tile_kf,
                    cond_image_strength=cond_image_strength,
                    neg_idx_latents=tile_neg_idx,
                    neg_idx=neg_idx,
                    neg_idx_strength=optional_negative_index_strength,
                    optional_positive_conditionings=optional_positive_conditionings,
                    guiding_start_step=guiding_start_step,
                    guiding_end_step=guiding_end_step,
                    guiding_latents=tile_guide,
                    guiding_strength=guiding_strength,
                    keyframe_per_tile_indices=kf_per_tile,
                    optional_keyframes=tile_kf,
                    first_seed=first_seed,
                    v=v, h=h,
                    vert_tiles=vertical_tiles, horiz_tiles=horizontal_tiles,
                    per_tile_seed_offsets=per_tile_seed_offsets_list,
                )

                # accumulate video with spatial weights
                tv = tile_video_out["samples"].to(video_samples.device)
                if video_final is None:
                    T_out = tv.shape[2]
                    video_final   = torch.zeros(B, C, T_out, height, width,
                                                device=video_samples.device,
                                                dtype=video_samples.dtype)
                    video_weights = torch.zeros_like(video_final)

                tw_weights = self._create_spatial_weights(
                    tv.shape, v, h, vertical_tiles, horizontal_tiles,
                    spatial_overlap, video_final.device, video_final.dtype,
                )
                video_final[:, :, :, vs:ve, hs:he]   += tv * tw_weights
                video_weights[:, :, :, vs:ve, hs:he] += tw_weights

                if v == 0 and h == 0 and tile_audio_out is not None:
                    audio_final = tile_audio_out

        video_final = video_final / (video_weights + 1e-8)
        out = {"samples": self._combine_av(video_final, audio_final, ltxav)}
        noise.seed = first_seed
        return (out,)

    def _create_spatial_weights(self, tile_shape, v, h,
                                 vertical_tiles, horizontal_tiles,
                                 spatial_overlap, device, dtype):
        w = torch.ones(tile_shape, device=device, dtype=dtype)
        if h > 0:
            b = torch.linspace(0, 1, spatial_overlap, device=device, dtype=dtype)
            w[:, :, :, :, :spatial_overlap] *= b.view(1, 1, 1, 1, -1)
        if h < horizontal_tiles - 1:
            b = torch.linspace(1, 0, spatial_overlap, device=device, dtype=dtype)
            w[:, :, :, :, -spatial_overlap:] *= b.view(1, 1, 1, 1, -1)
        if v > 0:
            b = torch.linspace(0, 1, spatial_overlap, device=device, dtype=dtype)
            w[:, :, :, :spatial_overlap, :] *= b.view(1, 1, 1, -1, 1)
        if v < vertical_tiles - 1:
            b = torch.linspace(1, 0, spatial_overlap, device=device, dtype=dtype)
            w[:, :, :, -spatial_overlap:, :] *= b.view(1, 1, 1, -1, 1)
        return w


NODE_CLASS_MAPPINGS = {
    "LTXVAVLoopingSampler": LTXVAVLoopingSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVAVLoopingSampler": "LTX AV Looping Sampler",
}
