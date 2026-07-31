"""
Per-speaker IC-LoRA references for the AV looping sampler.

The visual counterpart to LTXAVReferenceAudioMulti. That node banks one
reference AUDIO per speaker and the sampler picks by [SPEAKER n]; this one
banks reference IMAGES the same way, so a chunk tagged [SPEAKER 2] is
conditioned on speaker 2's views and not on everyone's at once.

Two reasons that matters beyond tidiness:

  - Identity. With every character's views present in every chunk, the prompt
    is the only thing arbitrating who is on screen, and a wardrobe phrase is a
    weak handle against a face the model can see. Sending only one character's
    views removes the ambiguity instead of describing around it.
  - Cost. References are appended to EVERY chunk, so their token cost is paid
    per chunk. Two characters at 4 views each plus a background is 9 views
    every chunk; scheduled, it is 5.

`all_chunks_images` is the exception that proves the rule — a background plate
belongs in every chunk regardless of who is speaking, so it banks under slot 0
and is merged into whichever speaker is selected. It is placed LAST in the
merged stack, keeping the tail slot a hand-built batch conventionally gives it
(MSR is positional, so this is not cosmetic).
"""

import node_helpers

from .ic_reference import IC_PACK_KEY, encode_reference_pack

IC_BANK_KEY = "ic_reference_bank"

MAX_SPEAKERS = 4


class LTXAVAddICReferencesMulti:
    """Bank reference views per speaker; the looping sampler selects per chunk."""

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "reference_images_1": ("IMAGE", {
                    "tooltip": "Views for [SPEAKER 1]. Injected only on chunks "
                               "whose prompt carries that tag."}),
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
            "optional": {
                "all_chunks_images": ("IMAGE", {
                    "tooltip": "Views injected on EVERY chunk regardless of speaker "
                               "— the background plate. Merged after the selected "
                               "speaker's views, so it keeps the tail slot."}),
            },
        }
        for i in range(2, MAX_SPEAKERS + 1):
            inputs["optional"][f"reference_images_{i}"] = ("IMAGE", {
                "tooltip": f"Views for [SPEAKER {i}]."})
        return inputs

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "attach"
    CATEGORY = "LTXAVTools/IC-LoRA"
    DESCRIPTION = (
        "Multi-speaker IC-LoRA references. Views for speaker n are injected only "
        "on chunks tagged [SPEAKER n] by LTXAVSpeakerPromptProvider; "
        "all_chunks_images go into every chunk. Requires the AV Looping Sampler — "
        "the bank is inert to any other sampler."
    )

    def attach(self, positive, negative, vae, reference_images_1, width, height,
               strength, latent_downscale_factor, layout, crop,
               all_chunks_images=None, **kwargs):
        bank = {}

        if all_chunks_images is not None:
            bank[0] = encode_reference_pack(
                vae, all_chunks_images, width, height, strength,
                latent_downscale_factor, layout, crop,
                label="LTXAVAddICReferencesMulti", slot="all chunks: ")

        slots = [reference_images_1] + [
            kwargs.get(f"reference_images_{i}") for i in range(2, MAX_SPEAKERS + 1)
        ]
        for n, images in enumerate(slots, start=1):
            if images is not None:
                bank[n] = encode_reference_pack(
                    vae, images, width, height, strength,
                    latent_downscale_factor, layout, crop,
                    label="LTXAVAddICReferencesMulti", slot=f"speaker {n}: ")

        speakers = sorted(k for k in bank if k)
        print(f"[LTXAVAddICReferencesMulti] banked speakers: {speakers}"
              + (" + all-chunks views" if 0 in bank else "")
              + ". Selected per chunk by [SPEAKER n].")
        if len(speakers) < 2 and 0 not in bank:
            print("[LTXAVAddICReferencesMulti] only one speaker is banked — this "
                  "behaves like the single Add IC References node. Wire "
                  "reference_images_2 (and up) to actually schedule.")

        positive = node_helpers.conditioning_set_values(
            positive, {IC_BANK_KEY: bank, IC_PACK_KEY: None})
        return (positive, negative)


NODE_CLASS_MAPPINGS = {"LTXAVAddICReferencesMulti": LTXAVAddICReferencesMulti}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXAVAddICReferencesMulti": "LTX AV Add IC References (Multi)"
}
