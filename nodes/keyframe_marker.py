"""
Keyframe position-embedding marking for appended guide tokens (LTX 2.5+).

LTX 2.5 adds a learned parameter `keyframes_abs_pos_embedding` that the model
adds to the tokens encoding a single standalone pixel frame
(comfy/ldm/lightricks/model.py, apply_keyframes_abs_pos_embedding). The mask
selecting those tokens marks the main sequence's first latent frame plus any
`generated_keyframes` slots, then explicitly CLEARS every appended guide token:

    mask = temporal_start == 0
    if num_guide_tokens > 0:
        mask[:, -num_guide_tokens:] = False      # model.py:1195-1196

An image keyframe added with LTXVAddGuide at a non-zero frame index arrives as
an appended guide token, so it never receives the marker. A keyframe at index 0
is a different case: the looping sampler writes it straight into latent frame 0
rather than appending it, so it lands on a token the mask does select and is
marked for free.

This node re-enables the marker for appended guide tokens.

LTX 2.3 and earlier have no such parameter. apply_keyframes_abs_pos_embedding
returns early when it is None, so the patched function is never called and this
node is inert on those checkpoints in every mode.
"""

import torch


class LTXAVKeyframeMarker:
    """Marks appended guide tokens with the LTX 2.5 keyframe position embedding."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["off", "single_frame_guides", "all_guides"], {
                    "default": "off",
                    "tooltip": "off — no patch applied; the model is returned "
                               "unchanged.\n"
                               "single_frame_guides — marks appended guide tokens "
                               "whose temporal extent is one pixel frame (image "
                               "keyframes), leaving multi-frame guides such as "
                               "IC-LoRA references and overlap carries unmarked.\n"
                               "all_guides — marks every appended guide token "
                               "regardless of temporal extent.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION     = "apply"
    CATEGORY     = "LTXAVTools/sampling"
    DESCRIPTION  = (
        "Adds the LTX 2.5 keyframe position embedding to appended guide tokens, "
        "which the model's own mask excludes. Takes a MODEL, returns a patched "
        "clone (source untouched). Defaults to off. No effect on LTX 2.3 and "
        "earlier, which have no keyframe position embedding."
    )

    def apply(self, model, mode):
        m = model.clone()
        if mode == "off":
            return (m,)

        # Read through the clone so stacking this node (or any other object
        # patch on the same attribute) composes instead of clobbering.
        orig = m.get_model_object("diffusion_model.keyframes_abs_pos_mask")
        mark_all = mode == "all_guides"
        warned = False

        def patched(pixel_coords, orig_shape, grid_mask, num_guide_tokens,
                    generated_keyframes):
            nonlocal warned
            mask = orig(pixel_coords, orig_shape, grid_mask, num_guide_tokens,
                        generated_keyframes)
            if num_guide_tokens <= 0:
                return mask

            if mark_all:
                mask[:, -num_guide_tokens:] = True
                return mask

            # pixel_coords dims: (batch, spatial [t,h,w], token, [start, end]).
            # SymmetricPatchifier is built with start_end=True everywhere in the
            # LTX path, so the last dim is normally present; a token spanning one
            # pixel frame has end - start == 1.
            temporal = pixel_coords[:, 0, -num_guide_tokens:, ...]
            if temporal.ndim == 3:
                single = (temporal[..., 1] - temporal[..., 0]) <= 1
            else:
                # No start/end coords: extent is unknowable here, so this mode
                # cannot discriminate and falls back to marking every guide.
                if not warned:
                    print("[LTXAVKeyframeMarker] pixel_coords carry no "
                          "start/end pair — cannot measure temporal extent, "
                          "marking all guide tokens (same as all_guides).")
                    warned = True
                single = torch.ones_like(temporal, dtype=torch.bool)

            mask[:, -num_guide_tokens:] |= single
            return mask

        m.add_object_patch("diffusion_model.keyframes_abs_pos_mask", patched)
        print(f"[LTXAVKeyframeMarker] mode={mode}")
        return (m,)


NODE_CLASS_MAPPINGS = {
    "LTXAVKeyframeMarker": LTXAVKeyframeMarker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXAVKeyframeMarker": "LTX AV Keyframe Marker (2.5)",
}
