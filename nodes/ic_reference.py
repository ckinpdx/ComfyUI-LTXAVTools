"""
Out-of-band IC-LoRA video references for the AV looping sampler.

See SPEC_IC_REFERENCES.md. The short version:

  - This node encodes reference views ONCE and attaches them to the positive
    conditioning as an inert `ic_reference_pack` key. It builds no guide tokens.
  - The LOOPING SAMPLER calls `apply_ic_references()` per chunk, after that
    chunk's conditioning is assembled, to turn the pack into real guide tokens.

That inversion is the whole point (SPEC §2.4): upstream-built tokens cannot
survive per-chunk conditioning replacement, which is exactly how `ref_audio`
used to get silently dropped by MultiPromptProvider. Injecting after assembly
is multiprompt-safe by construction.

Consequence worth stating plainly: **wiring this node into a stock
SamplerCustomAdvanced does nothing.** Nothing there calls apply_ic_references(),
so the pack rides along ignored. That is the design, not a bug.
"""

import torch
import comfy.utils
import node_helpers

IC_PACK_KEY = "ic_reference_pack"

# LTX temporal VAE: latent frame 0 covers 1 pixel frame, every latent frame
# after it covers 8. Encoding truncates the input to 8n+1 frames.
_TIME_SCALE = 8


def _strip_pack(cond):
    """Remove the pack key. Inert to the model, but dropping it after use keeps
    a re-used conditioning from being injected twice."""
    if cond is None:
        return None
    return node_helpers.conditioning_set_values(cond, {IC_PACK_KEY: None})


def get_ic_pack(cond):
    if not cond:
        return None
    for t in cond:
        pack = t[1].get(IC_PACK_KEY)
        if pack is not None:
            return pack
    return None


def build_reference_stack(images, layout):
    """
    Arrange reference views into a pixel stack the temporal VAE will keep whole.

    This is the trap this node exists to defuse. The encoder truncates to
    `((N - 1) // 8) * 8 + 1` frames, so handing it 4 stacked views keeps
    ((4-1)//8)*8+1 = **1** — three references silently discarded, no error, and
    a run that looks merely mediocre rather than broken.

    `one_latent_per_view` lays out [v0, v1*8, v2*8, ... ] = 8*(N-1)+1 frames,
    which is exactly 8n+1, so nothing is dropped and each view lands on its own
    latent frame (frame 0 -> latent 0; each following 8-frame group -> 1 latent).

    `as_sequence` is for input that is ALREADY a correctly-built temporal
    multiplex — notably the Licon MSR node's reference video, which lays subjects
    onto whole latent frames with the background last. Re-stacking that would
    destroy the structure it was built to have, so this layout encodes it as-is
    and only checks the 8n+1 alignment.
    """
    n = images.shape[0]
    if layout == "as_sequence":
        keep = ((n - 1) // _TIME_SCALE) * _TIME_SCALE + 1
        if keep != n:
            print(f"[LTXAVAddICReferences] as_sequence: {n} frames is not 8n+1 — "
                  f"the VAE keeps {keep} and drops {n - keep}. Feed a sequence built "
                  f"on latent boundaries (17, 25, 33, 41, 49, 57, 65 ...).")
        return images[:keep], (keep - 1) // _TIME_SCALE + 1
    if layout == "single_frame" or n == 1:
        return images[:1], 1
    frames = [images[:1]]
    for i in range(1, n):
        frames.append(images[i:i + 1].repeat(_TIME_SCALE, 1, 1, 1))
    return torch.cat(frames, dim=0), n


class LTXAVAddICReferences:
    """Encode reference views and attach them as an inert pack (see module doc)."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "reference_images": ("IMAGE", {
                    "tooltip": "Reference views. With layout=one_latent_per_view "
                               "each image gets its own latent frame."}),
                "width": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 32,
                    "tooltip": "GENERATION pixel width (not the reference's). "
                               "Must match the sampler's latent or apply-time errors."}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 32,
                    "tooltip": "GENERATION pixel height."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "IC convention is 1.0. Below 1.0 is known to bleed."}),
                "latent_downscale_factor": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 10.0, "step": 1.0,
                    "tooltip": "From the LoRA's reference_downscale_factor metadata "
                               "(LTX LoRA Metadata Reader / LTXICLoRALoaderModelOnly). "
                               "1 = full grid, 2 = half, ..."}),
                "layout": (["one_latent_per_view", "as_sequence", "single_frame"], {
                    "default": "one_latent_per_view",
                    "tooltip": "one_latent_per_view: N views -> N latent frames "
                               "(pads to 8n+1 so none are dropped). "
                               "as_sequence: input is ALREADY a temporal multiplex "
                               "(e.g. Licon MSR's reference video) — encode as-is. "
                               "single_frame: first image only — use for a "
                               "pre-composited reference sheet."}),
                "crop": (["disabled", "center"], {"default": "disabled"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "attach"
    CATEGORY = "LTXAVTools/IC-LoRA"

    def attach(self, positive, negative, vae, reference_images, width, height,
               strength, latent_downscale_factor, layout, crop):
        factor = max(1, int(round(float(latent_downscale_factor))))

        ts = vae.downscale_index_formula          # (time, width, height) scales
        _, w_scale, h_scale = ts
        lat_w = width // w_scale
        lat_h = height // h_scale

        # Mirrors the official node's guard: dilation needs the full grid to
        # partition evenly, and the failure downstream is an opaque shape error.
        if factor > 1 and (lat_w % factor or lat_h % factor):
            raise ValueError(
                f"[LTXAVAddICReferences] generation latent grid {lat_w}x{lat_h} "
                f"(from {width}x{height}) is not divisible by "
                f"latent_downscale_factor {factor}. Pick generation dims whose "
                f"latent grid divides evenly — e.g. a multiple of {32 * factor} px."
            )

        n_views = reference_images.shape[0]
        stack, kept = build_reference_stack(reference_images, layout)
        if layout == "single_frame" and n_views > 1:
            print(f"[LTXAVAddICReferences] layout=single_frame — using the first of "
                  f"{n_views} images. Use one_latent_per_view to keep them all.")

        target_w = int(lat_w * w_scale / factor)
        target_h = int(lat_h * h_scale / factor)
        pixels = comfy.utils.common_upscale(
            stack.movedim(-1, 1), target_w, target_h, "bilinear", crop=crop
        ).movedim(1, -1)[:, :, :, :3]

        latents = vae.encode(pixels)

        # The truncation this node exists to defuse — assert it did not happen.
        if latents.shape[2] != kept:
            raise RuntimeError(
                f"[LTXAVAddICReferences] encoded {latents.shape[2]} latent frames "
                f"from {kept} reference view(s) — expected {kept}. The temporal VAE "
                f"truncates to 8n+1 pixel frames; the stack was {stack.shape[0]}."
            )

        pack = {
            "latents": latents.cpu(),
            "strength": float(strength),
            "factor": factor,
            "gen_latent_hw": (lat_h, lat_w),
            "n_views": kept,
        }
        positive = node_helpers.conditioning_set_values(positive, {IC_PACK_KEY: pack})

        # Two counts, and confusing them is easy: at factor > 1 the guide is
        # DILATED back onto the full grid, so append_keyframe reports
        # frames*lat_h*lat_w keyframe tokens ("pre-filter"), of which only every
        # factor-th position is real — the holes carry mask -1 and the model's
        # grid_mask drops them. The real attention cost is the smaller number.
        real = latents.shape[2] * (lat_h // factor) * (lat_w // factor)
        pre = latents.shape[2] * lat_h * lat_w
        print(f"[LTXAVAddICReferences] {kept} reference view(s) -> "
              f"{latents.shape[2]} latent frames at {latents.shape[4]}x{latents.shape[3]} "
              f"(factor {factor}), strength {strength}. ~{real} tokens per chunk"
              + (f" ({pre} pre-filter, holes dropped by grid_mask)" if factor > 1 else "")
              + ". Inert until the looping sampler injects it.")
        return (positive, negative)


def apply_ic_references(add_guide_fn, vae, positive, negative,
                        video_latent_dict, pack, chunk_index=0):
    """
    Turn a pack into real guide tokens on one chunk.

    `add_guide_fn` is the sampler's bound `_add_latent_guide`, injected rather
    than reimplemented: that method already mirrors the official convention
    (dilation, append_keyframe, and the MEASURED pre_filter_count read from the
    conditioning delta rather than a predicted formula). Duplicating it here
    would mean two copies drifting against core's frame accounting, which has
    already changed under this pack once.

    References sit at frame_idx 0 — out-of-band tokens whose RoPE coords place
    them at the timeline start, appended past the end of the latent tensor.

    Returns (positive, negative, video_latent_dict) with the pack key stripped.
    """
    if pack is None:
        return positive, negative, video_latent_dict

    lat_h, lat_w = pack["gen_latent_hw"]
    cur_h, cur_w = video_latent_dict["samples"].shape[3:5]
    if (cur_h, cur_w) != (lat_h, lat_w):
        raise ValueError(
            f"[LTXAVAddICReferences] reference pack was built for a "
            f"{lat_w}x{lat_h} latent grid but this chunk is {cur_w}x{cur_h}. "
            f"Set the node's width/height to the generation resolution."
        )

    guide = {"samples": pack["latents"].to(video_latent_dict["samples"].device,
                                           dtype=video_latent_dict["samples"].dtype)}
    positive, negative, video_latent_dict = add_guide_fn(
        vae, positive, negative, video_latent_dict, guide,
        latent_idx=0, strength=pack["strength"],
        downscale_factor=pack["factor"],
    )
    print(f"[LTXAVAddICReferences] chunk {chunk_index}: injected "
          f"{pack['n_views']} reference view(s).")
    return _strip_pack(positive), _strip_pack(negative), video_latent_dict


NODE_CLASS_MAPPINGS = {"LTXAVAddICReferences": LTXAVAddICReferences}
NODE_DISPLAY_NAME_MAPPINGS = {"LTXAVAddICReferences": "LTX AV Add IC References"}
