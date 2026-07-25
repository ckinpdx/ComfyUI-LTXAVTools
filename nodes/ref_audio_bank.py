"""
Carry-swap reference audio for per-chunk voice identity (SPEC_NEG_REF_AUDIO.md).

LTXAVReferenceAudioBank encodes up to four reference voices as raw audio
latents and bundles them with a per-chunk voice schedule. The AV Looping
Sampler (optional_ref_audio_bank input) substitutes the scheduled voice into
the extend chunk's frozen audio carry slot — "reference audio as though it was
from a prior chunk." The real accumulator tail is never modified and the
keep-verbatim stitch keeps it in the output; the ref is sampling context only.

This rides the proven voice-transfer pathway (frozen on-timeline audio prefix)
and needs no ID-LoRA. Swap only occurs at scripted gaps (see the spec's
turn-seam choreography): a mid-speech swap creates an output seam
discontinuity because the new region is generated following the ref's tail
but spliced after the accumulator's real tail.
"""

import torch
import torchaudio


def _encode_ref_latent(audio_vae, reference_audio):
    """AUDIO dict -> raw audio latent [B, C, T, F] (NOT ref_audio tokens)."""
    sample_rate = reference_audio["sample_rate"]
    vae_sample_rate = getattr(audio_vae, "audio_sample_rate", 44100)
    waveform = reference_audio["waveform"]
    if vae_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, vae_sample_rate)
    return audio_vae.encode(waveform.movedim(1, -1))


def _silence_latent(audio_vae, n_frames, like_waveform):
    """N frames of VAE-encoded real silence [B, C, N, F] — NOT a zero latent
    (zeros are OOD for a2v attention and decode to a faint hum). Encodes a
    zero WAVEFORM and takes a stable interior slice (avoids the encoder's
    edge frames). Channels/format match the reference waveform."""
    sr = getattr(audio_vae, "audio_sample_rate", 44100)
    channels = like_waveform.shape[1]
    samples = max(1, int(round(((n_frames + 8) / 25.0) * sr)))
    wav = torch.zeros(1, channels, samples, dtype=torch.float32)
    lat = audio_vae.encode(wav.movedim(1, -1))
    if lat.shape[2] >= n_frames + 4:
        return lat[:, :, 4:4 + n_frames, :].contiguous()
    return lat[:, :, :n_frames, :].contiguous()


class LTXAVReferenceAudioBank:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_vae": ("VAE", {"tooltip": "LTXV audio VAE."}),
                "reference_audio_1": ("AUDIO", {
                    "tooltip": "Voice 1. A clean single-speaker clip; it should "
                               "cover at least the audio overlap span (~2 s is "
                               "comfortable at overlap 48). The first overlap-length "
                               "frames are used.",
                }),
                "schedule": ("STRING", {
                    "default": "1",
                    "tooltip": "Pipe/comma-separated voice number per chunk, aligned "
                               "1:1 with the prompt segments / scene_lengths "
                               "(e.g. '1|2|1|2' for alternating turns). Shorter than "
                               "the chunk count -> last entry repeats. Chunk 0 never "
                               "swaps (no carry slot); its entry exists for alignment "
                               "and for deciding whether chunk 1 is a voice change.",
                }),
                "swap_mode": (["on_change", "always"], {
                    "default": "on_change",
                    "tooltip": "on_change: swap the audio carry only when the "
                               "schedule's voice differs from the previous chunk "
                               "(turn seams). always: swap every extend chunk — for "
                               "testing/anti-drift; only sane when every seam lands "
                               "in a scripted gap, since a mid-speech swap causes an "
                               "output discontinuity at the splice.",
                }),
                "trailing_silence_frames": ("INT", {
                    "default": 0, "min": 0, "max": 64,
                    "tooltip": "Latent frames of VAE-encoded silence the sampler "
                               "places at the END of the swapped carry: the ref voice "
                               "sits at the front (identity established), the last N "
                               "frames are quiet so the new region onsets from silence "
                               "instead of continuing a mid-word ref tail. 0 = off "
                               "(carry is the ref's front only). ~1-3 frames (~40-120ms) "
                               "is a clean seam; more silence weakens transfer. Applied "
                               "in the sampler where the overlap length is known, so it "
                               "works regardless of ref length.",
                }),
            },
            "optional": {
                "reference_audio_2": ("AUDIO",),
                "reference_audio_3": ("AUDIO",),
                "reference_audio_4": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("LTXAV_REF_BANK",)
    RETURN_NAMES = ("ref_audio_bank",)
    FUNCTION     = "build"
    CATEGORY     = "LTXAVTools/audio"
    DESCRIPTION  = (
        "Encodes up to four reference voices and a per-chunk schedule for "
        "carry-swap voice identity. Wire to the AV Looping Sampler's "
        "optional_ref_audio_bank. The scheduled voice replaces the frozen audio "
        "carry of extend chunks (sampling context only — never appears in the "
        "output). No ID-LoRA required. See SPEC_NEG_REF_AUDIO.md."
    )

    def build(self, audio_vae, reference_audio_1, schedule, swap_mode,
              trailing_silence_frames=0,
              reference_audio_2=None, reference_audio_3=None, reference_audio_4=None):

        latents = {}
        for n, ref in enumerate(
            [reference_audio_1, reference_audio_2, reference_audio_3, reference_audio_4],
            start=1,
        ):
            if ref is not None:
                lat = _encode_ref_latent(audio_vae, ref)
                latents[n] = lat
                print(f"[LTXAVReferenceAudioBank] voice {n}: {lat.shape[2]} audio latents "
                      f"(~{lat.shape[2] / 25.0:.2f}s)")

        parsed = []
        for i, tok in enumerate(schedule.replace(",", "|").split("|")):
            tok = tok.strip()
            try:
                voice = int(tok)
            except ValueError:
                voice = 1
                if tok:
                    print(f"[LTXAVReferenceAudioBank] schedule entry {i} ('{tok}') "
                          "invalid — using voice 1.")
            if voice not in latents:
                print(f"[LTXAVReferenceAudioBank] schedule entry {i} names voice "
                      f"{voice} but no reference_audio_{voice} is connected — "
                      "using voice 1.")
                voice = 1
            parsed.append(voice)
        if not parsed:
            parsed = [1]

        trailing_silence = None
        n_sil = max(0, int(trailing_silence_frames))
        if n_sil > 0:
            trailing_silence = _silence_latent(
                audio_vae, n_sil, reference_audio_1["waveform"])
            print(f"[LTXAVReferenceAudioBank] trailing silence: {n_sil} frames "
                  f"(~{n_sil / 25.0 * 1000:.0f}ms) appended to each swapped carry tail")

        print(f"[LTXAVReferenceAudioBank] schedule: {parsed} | swap_mode: {swap_mode}")

        return ({
            "latents": latents,
            "schedule": parsed,
            "swap_mode": swap_mode,
            "trailing_silence": trailing_silence,
        },)


NODE_CLASS_MAPPINGS = {
    "LTXAVReferenceAudioBank": LTXAVReferenceAudioBank,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXAVReferenceAudioBank": "LTX AV Reference Audio Bank (Carry-Swap)",
}
