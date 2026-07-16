"""
Cross-modal attention toggles for the LTX 2.3 AV model.

The AV transformer block runs two cross-modal couplings each step, both gated by
transformer_options (see comfy/ldm/lightricks/av_model.py:267-268):

    run_a2v = ... transformer_options.get("a2v_cross_attn", True)   # audio -> video
    run_v2a = ... transformer_options.get("v2a_cross_attn", True)   # video -> audio

This node exposes those gates so you can switch a coupling off for a generation.

Primary use — regenerating/changing speech over a GUIDE video:
    When you generate audio while a guide video's lips articulate DIFFERENT words
    than your text, the video->audio (v2a) path reads those visible lips and drags
    the emerging audio toward the ORIGINAL words. In training, lips<->audio is a
    near-deterministic pairing while text->audio is one-to-many, so the crisp lip
    signal out-muscles the diffuse text signal and the audio never resolves into
    coherent speech — you get gibberish. Set v2a_cross_attn = False and the audio
    is driven purely by the text (the new words); a2v still runs, so the video's
    mouth syncs to the newly generated audio.

    v2a only bites during audio GENERATION. When audio is supplied as input
    (frozen), v2a is irrelevant, which is why input-audio lipsync already works.
"""

import comfy.model_patcher  # noqa: F401  (documents where the gates are read)


class LTXAVCrossAttnToggle:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "v2a_cross_attn": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Video->audio coupling. Default ON. Turn OFF when "
                               "GENERATING audio over a guide video whose lips say "
                               "DIFFERENT words than your text — otherwise the "
                               "guide's visible lips dominate the text and corrupt "
                               "generated speech into gibberish. OFF = audio driven "
                               "purely by the text prompt; a2v still syncs the video "
                               "mouth to the new audio. No effect when audio is "
                               "supplied as input (frozen).",
                }),
                "a2v_cross_attn": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Audio->video coupling (this IS lipsync — the mouth "
                               "following the audio). Default ON. Turn OFF only to "
                               "fully decouple video from audio; disabling it kills "
                               "lipsync. Rarely wanted.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION     = "apply"
    CATEGORY     = "LTXAVTools/audio"
    DESCRIPTION  = (
        "Toggle the LTX 2.3 AV model's cross-modal attention couplings "
        "(audio<->video). Turn v2a off to generate new-word speech over a guide "
        "video without the guide's lips corrupting the audio into gibberish."
    )

    def apply(self, model, v2a_cross_attn, a2v_cross_attn):
        # clone() deep-copies model_options (model_patcher.py:405), and
        # model_options always carries a transformer_options dict (line 308), so
        # setting the gates here does not touch the source model.
        m = model.clone()
        to = m.model_options["transformer_options"]
        to["v2a_cross_attn"] = bool(v2a_cross_attn)
        to["a2v_cross_attn"] = bool(a2v_cross_attn)
        print(f"[LTXAVCrossAttnToggle] v2a_cross_attn={bool(v2a_cross_attn)} "
              f"a2v_cross_attn={bool(a2v_cross_attn)}")
        return (m,)


NODE_CLASS_MAPPINGS = {
    "LTXAVCrossAttnToggle": LTXAVCrossAttnToggle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXAVCrossAttnToggle": "LTX AV Cross-Attention Toggle",
}
