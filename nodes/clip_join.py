"""
Clip joining and bridge preparation for LTX AV latents.

Two nodes:

  LTX AV Join Latents  — concatenate finished AV clips correctly.
  LTX AV Bridge Prep   — stage the inputs for a generated transition between two
                         clips (the sampler does the actual bridging).

The whole reason these exist is the LTX first-frame asymmetry. Latent 0 of a
sequence decodes to ONE pixel frame; every latent after it decodes to eight. So
a naive `torch.cat` of two clips is wrong by a CONSTANT amount, every time:

    naive video: +7 pixel frames   (clip B's latent 0 was encoded as a video
                                    start holding 1 frame, but once it sits mid
                                    sequence it decodes as 8)
    naive audio: +1 audio latent   (audio_pos telescopes; B's own count is one
                                    more than the span it should contribute)

Both are fixed by dropping each subsequent clip's stub video latent and its
first audio latent. That is also exactly the "first frame flash" everyone
trims by hand — here it is exact instead of eyeballed.
"""

import torch
from comfy.nested_tensor import NestedTensor

from .utils import audio_pos, infer_av_fps


def _split_av(latent, label):
    """AV NestedTensor -> (video [B,C,T,H,W], audio [B,C,T,F])."""
    samples = latent["samples"]
    if not isinstance(samples, NestedTensor):
        raise ValueError(
            f"[{label}] expects an AV NestedTensor latent (video + audio). "
            f"Got a plain latent — use LTXVConcatAVLatent, or the AV VAE encode."
        )
    return samples.tensors[0], samples.tensors[1]


def _clip_fps_check(v, a, fps, label, name):
    """Verify a clip's own video:audio ratio agrees with the fps you declared.

    Audio is 25 latents/second regardless of video rate, so audio_pos(T_v, fps)
    == T_a pins the rate — the clip can be ASKED what it is rather than trusted.
    This is the cheapest possible guard against the session's recurring failure:
    a rate mismatch that never errors and just scales everything by 2.
    """
    T_v, T_a = v.shape[2], a.shape[2]
    want = audio_pos(T_v, fps)
    if T_a == want:
        return
    guess = infer_av_fps(T_v, T_a)
    if guess:
        raise ValueError(
            f"[{label}] {name} has {T_v} video / {T_a} audio latents, which is "
            f"{' or '.join(f'{g:g}' for g in guess)} fps — but fps is set to "
            f"{fps:g} (that would need {want} audio latents). Set fps to the rate "
            f"these clips were generated at."
        )
    raise ValueError(
        f"[{label}] {name} has {T_v} video / {T_a} audio latents, but "
        f"audio_pos({T_v}, {fps:g}) = {want}. The audio does not match the video "
        f"at any standard rate — the clip's streams are out of sync."
    )


def _retime_audio(a, want):
    """Resample an audio latent sequence along time to `want` frames.

    Declaring a clip at a lower fps means its video plays slower, so it covers
    more wall-clock and needs proportionally MORE audio. Linear interpolation of
    the latent sequence is the latent-domain equivalent of a time stretch. For
    the 24<->25 case that is a ~4% change; it is an approximation, not a proper
    resampler, so it is reported whenever it happens.
    """
    if a.shape[2] == want:
        return a
    B, C, T, F = a.shape
    flat = a.permute(0, 1, 3, 2).reshape(B, C * F, T)
    out = torch.nn.functional.interpolate(flat, size=want, mode="linear",
                                          align_corners=False)
    return out.reshape(B, C, F, want).permute(0, 1, 3, 2).contiguous()


def _resolve_fps(parts, fps, label):
    """Decide the output rate and bring every clip onto it.

    fps <= 0 means AUTO: ask each clip what rate it is (its video:audio ratio
    pins it) and take the LOWEST, since that is the rate every clip can be
    expressed at without dropping video latents. Video is never touched — a
    clip is T latents whatever rate you call it — only the audio is retimed.
    """
    inferred = []
    for i, (v, a) in enumerate(parts, start=1):
        guess = infer_av_fps(v.shape[2], a.shape[2])
        inferred.append(guess[0] if guess else None)

    if fps and fps > 0:
        for i, (v, a) in enumerate(parts, start=1):
            _clip_fps_check(v, a, fps, label, f"clip {i}")
        return float(fps), parts

    unknown = [i + 1 for i, f in enumerate(inferred) if f is None]
    if unknown:
        raise ValueError(
            f"[{label}] fps is set to auto but clip(s) {unknown} match no standard "
            f"rate — their video:audio ratio is not a valid AV latent pair. Set fps "
            f"explicitly, or check those clips."
        )

    target = float(min(inferred))
    out = []
    for i, ((v, a), native) in enumerate(zip(parts, inferred), start=1):
        want = audio_pos(v.shape[2], target)
        if a.shape[2] != want:
            # Declaring a clip at a LOWER rate does not shorten it — the video
            # latent count is fixed, so it plays LONGER. Retiming the audio keeps
            # sync, but the clip genuinely runs slower. At 24<->25 that is 4% and
            # nobody notices; at 25<->50 it is HALF SPEED, which is never what was
            # meant — that needs a temporal video resample, which this node
            # cannot do. Say so with the actual durations rather than hiding it.
            slow = native / target
            px = (v.shape[2] - 1) * 8 + 1
            msg = (f"[{label}] clip {i}: {native:g} -> {target:g} fps, audio "
                   f"retimed {a.shape[2]} -> {want} latents "
                   f"({(want / a.shape[2] - 1) * 100:+.1f}%).")
            if slow > 1.10:
                print(msg)
                print(f"    *** WARNING: this clip now plays {slow:.2f}x SLOWER — "
                      f"{px / native:.2f}s becomes {px / target:.2f}s. Video latent "
                      f"count is fixed, so a lower rate stretches duration. To keep "
                      f"its real speed you must temporally resample the VIDEO "
                      f"(e.g. regenerate it at {target:g} fps, or use the temporal "
                      f"upsampler on the slower clip instead). ***")
            else:
                print(f"{msg} Plays {slow:.3f}x slower "
                      f"({px / native:.2f}s -> {px / target:.2f}s); video untouched, "
                      f"latent-domain stretch, not a resampler.")
            a = _retime_audio(a, want)
        out.append((v, a))
    rates = ", ".join(f"clip {i+1} {f:g}" for i, f in enumerate(inferred))
    print(f"[{label}] fps auto: detected {rates} -> output {target:g} fps.")
    return target, out


def _downscale_to(v, h, w):
    """Area-resample a video latent's spatial grid (downscale only)."""
    B, C, T, _, _ = v.shape
    flat = v.permute(0, 2, 1, 3, 4).reshape(B * T, C, v.shape[3], v.shape[4])
    out = torch.nn.functional.interpolate(flat, size=(h, w), mode="area")
    return out.reshape(B, T, C, h, w).permute(0, 2, 1, 3, 4).contiguous()


def _reconcile_sizes(videos, mode, label, aspect_tol=0.02):
    """Bring clips to a common latent grid.

    Joining clips of different RESOLUTION is the normal case — the same shot
    graded at two sizes, a stage-1 and a stage-2 output — so matching aspect
    ratios reconcile automatically by downscaling to the smallest. Differing
    ASPECT is a real incompatibility: there is no correct answer without
    cropping or letterboxing, both of which change the framing, so that errors.

    The target is the smallest clip's OWN grid rather than (min height, min
    width) independently, which could otherwise synthesise a grid matching no
    clip's aspect when latent dims round.
    """
    dims = [(v.shape[3], v.shape[4]) for v in videos]
    if len(set(dims)) == 1:
        return videos

    aspects = [w / h for h, w in dims]
    if max(aspects) / min(aspects) > 1.0 + aspect_tol:
        detail = ", ".join(f"clip {i+1} {w}x{h} ({w/h:.3f}:1)"
                           for i, (h, w) in enumerate(dims))
        raise ValueError(
            f"[{label}] aspect ratio mismatch ({detail}). Resolution differences "
            f"are reconciled automatically, but differing aspect cannot be — "
            f"cropping or letterboxing changes the framing, so that is your call. "
            f"Resize in pixel space to a common aspect and re-encode."
        )

    if mode == "error":
        detail = ", ".join(f"clip {i+1} {w}x{h}" for i, (h, w) in enumerate(dims))
        raise ValueError(
            f"[{label}] resolution mismatch ({detail}) and on_size_mismatch = "
            f"error. Set it to downscale_to_smallest to reconcile."
        )

    h, w = min(dims, key=lambda hw: hw[0] * hw[1])
    detail = ", ".join(f"{d[1]}x{d[0]}" for d in dims)
    print(f"[{label}] resolution mismatch ({detail}), matching aspect -> "
          f"downscaling all to {w}x{h} on the LATENT grid. Lossy: latents are not "
          f"images. Decode, resize and re-encode if the detail matters.")
    return [v if (v.shape[3], v.shape[4]) == (h, w) else _downscale_to(v, h, w)
            for v in videos]


def _check_compatible(v_ref, v, i, label):
    if v.shape[1] != v_ref.shape[1] or v.shape[3:] != v_ref.shape[3:]:
        raise ValueError(
            f"[{label}] clip {i} is {v.shape[4]}x{v.shape[3]} with {v.shape[1]} "
            f"channels but clip 1 is {v_ref.shape[4]}x{v_ref.shape[3]} with "
            f"{v_ref.shape[1]}. All clips must share resolution and channel count."
        )


class LTXAVJoinLatents:
    """Concatenate 2-4 AV latents end to end, correcting the first-frame overlap."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent_1": ("LATENT",),
                "latent_2": ("LATENT",),
                "fps": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 200.0, "step": 0.01,
                    "tooltip": "0 = AUTO: each clip's rate is read from its own "
                               "video:audio ratio and the LOWEST wins (e.g. 24 + 25 "
                               "-> 24), with the faster clips' audio retimed to "
                               "match. Video is never touched. Set a value to force "
                               "a rate — every clip is then validated against it.",
                }),
            },
            "optional": {
                "latent_3": ("LATENT",),
                "latent_4": ("LATENT",),
                "free_clips": ("STRING", {
                    "default": "",
                    "tooltip": "1-based clip indices to mark WHITE (regenerate) in "
                               "region_mask; everything else is black (keep). For a "
                               "bridge reassembly join A_tail|C|B_head, use '2' to "
                               "free the bridge. Empty = all-black mask.",
                }),
                "mask_feather": ("INT", {
                    "default": 0, "min": 0, "max": 64,
                    "tooltip": "Pixel frames of linear ramp on each side of a freed "
                               "region, so the refinement pass does not show a hard "
                               "mask edge at the seam.",
                }),
                "on_size_mismatch": (["downscale_to_smallest", "error"], {
                    "default": "downscale_to_smallest",
                    "tooltip": "Clips of different RESOLUTION but matching aspect "
                               "reconcile automatically to the smallest. Latent-grid "
                               "resampling is lossy — pixel-space resize + re-encode "
                               "keeps more detail. Set error to refuse instead. "
                               "Differing ASPECT always errors: cropping or "
                               "letterboxing changes framing, so that is your call.",
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "MASK", "STRING")
    RETURN_NAMES = ("latent", "region_mask", "info")
    FUNCTION = "join"
    CATEGORY = "LTXAVTools/latent"
    DESCRIPTION = (
        "Concatenates AV latents end to end with the LTX first-frame correction "
        "(a plain cat is wrong by +7 pixel frames and +1 audio latent per join). "
        "Also emits a temporal region_mask for refining a chosen clip in place."
    )

    def join(self, latent_1, latent_2, fps, latent_3=None, latent_4=None,
             free_clips="", mask_feather=0, on_size_mismatch="downscale_to_smallest"):
        clips = [c for c in (latent_1, latent_2, latent_3, latent_4) if c is not None]
        label = "LTXAVJoinLatents"

        # Ask each clip what rate it is before touching anything: a mismatch here
        # is silent downstream and scales the whole result.
        parts = [_split_av(c, label) for c in clips]
        fps, parts = _resolve_fps(parts, fps, label)
        vids = _reconcile_sizes([v for v, _ in parts], on_size_mismatch, label)
        clips = [{"samples": NestedTensor([vids[i], parts[i][1]])}
                 for i in range(len(parts))]

        v0, a0 = _split_av(clips[0], label)
        videos, audios, spans, cum_v = [v0], [a0], [], v0.shape[2]
        spans.append((0, v0.shape[2]))

        for i, c in enumerate(clips[1:], start=2):
            v, a = _split_av(c, label)
            _check_compatible(v0, v, i, label)
            if v.shape[2] < 2 or a.shape[2] < 2:
                raise ValueError(
                    f"[{label}] clip {i} has {v.shape[2]} video / {a.shape[2]} audio "
                    f"latents — too short to join (its stub latent must be dropped, "
                    f"leaving nothing)."
                )
            # Drop the stub: clip i's latent 0 was encoded as a video START (1 pixel
            # frame) and would decode as 8 once it sits mid-sequence. Its first audio
            # latent goes with it so the totals stay on the audio_pos map.
            v, a = v[:, :, 1:], a[:, :, 1:]
            spans.append((cum_v, cum_v + v.shape[2]))
            cum_v += v.shape[2]
            videos.append(v)
            audios.append(a)

        video = torch.cat(videos, dim=2)
        audio = torch.cat(audios, dim=2)
        T_v = video.shape[2]

        # Reconcile against the boundary map rather than trusting the inputs: a clip
        # whose own audio was short/long would otherwise poison the total silently.
        want_a = audio_pos(T_v, fps)
        got_a = audio.shape[2]
        if got_a != want_a:
            print(f"[{label}] audio {got_a} -> {want_a} to match audio_pos({T_v}, "
                  f"{fps:g}) (delta {got_a - want_a:+d}). If this is more than a "
                  f"frame or two, an input clip's audio did not match its video.")
            if got_a > want_a:
                audio = audio[:, :, :want_a]
            else:
                pad = torch.zeros(audio.shape[0], audio.shape[1], want_a - got_a,
                                  audio.shape[3], device=audio.device, dtype=audio.dtype)
                audio = torch.cat([audio, pad], dim=2)

        px = (T_v - 1) * 8 + 1
        mask = self._region_mask(spans, T_v, px, free_clips, mask_feather, label)

        info = (f"joined {len(clips)} clips -> {T_v} video latents ({px} px frames, "
                f"{px / fps:.2f}s @ {fps:g}fps), {audio.shape[2]} audio latents. "
                f"Dropped {len(clips) - 1} stub latent(s).")
        print(f"[{label}] {info}")

        return ({"samples": NestedTensor([video, audio])}, mask, info)

    @staticmethod
    def _region_mask(spans, T_v, px, free_clips, feather, label):
        """[px, 64, 64] MASK — white over freed clips, spatially uniform."""
        prof = torch.zeros(px, dtype=torch.float32)
        idxs = []
        for tok in free_clips.replace(",", "|").split("|"):
            tok = tok.strip()
            if tok:
                try:
                    idxs.append(int(tok))
                except ValueError:
                    print(f"[{label}] free_clips: ignoring non-numeric '{tok}'")
        for i in idxs:
            if not 1 <= i <= len(spans):
                print(f"[{label}] free_clips: clip {i} out of range 1..{len(spans)}")
                continue
            s_lat, e_lat = spans[i - 1]
            # latent -> pixel frame (latent 0 owns frame 0; latent k owns 8k-7..8k)
            s_px = 0 if s_lat == 0 else 8 * s_lat - 7
            e_px = min(px, 8 * e_lat - 7 if e_lat > 0 else 1)
            prof[s_px:e_px] = 1.0
            if feather > 0:
                ramp = torch.linspace(0, 1, feather + 2, dtype=torch.float32)[1:-1]
                head = min(feather, max(0, e_px - s_px) // 2)
                if s_px > 0 and head > 0:
                    prof[s_px:s_px + head] = ramp[:head]
                if e_px < px and head > 0:
                    prof[e_px - head:e_px] = ramp.flip(0)[:head]
        return prof.view(px, 1, 1).expand(px, 64, 64).contiguous()


class LTXAVBridgePrep:
    """Stage clip A + clip B for a generated transition between them."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_a": ("LATENT", {"tooltip": "The outgoing clip."}),
                "clip_b": ("LATENT", {"tooltip": "The incoming clip."}),
                "vae": ("VAE", {"tooltip": "Video VAE — decodes clip B's first frame."}),
                "bridge_seconds": ("FLOAT", {
                    "default": 1.0, "min": 0.08, "max": 20.0, "step": 0.04,
                    "tooltip": "Length of the transition to generate. Short is easier: "
                               "the model has to arrive at a fixed frame, and every "
                               "extra second is more freedom to drift before it does.",
                }),
                "a_tail_seconds": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 60.0, "step": 0.04,
                    "tooltip": "How much of clip A's tail to feed as prior context. "
                               "0 = all of A (the sampler then re-walks the whole "
                               "clip). The bridge only needs local context, so a "
                               "couple of seconds is normally plenty.",
                }),
                "fps": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 200.0, "step": 0.01,
                    "tooltip": "0 = AUTO: read each clip's rate from its own "
                               "video:audio ratio, lowest wins. The resolved rate is "
                               "in `info` and the plan — set the sampler's video_fps "
                               "and that pass's LTXVConditioning frame_rate to it.",
                }),
            },
        }

    # APPEND-ONLY: trim_prior_latents is last. ComfyUI links outputs by index.
    RETURN_TYPES = ("LATENT", "LATENT", "IMAGE", "STRING", "STRING", "INT")
    RETURN_NAMES = ("prior_av_latent", "bridge_latent", "end_image",
                    "cond_indices", "info", "trim_prior_latents")
    FUNCTION = "prep"
    CATEGORY = "LTXAVTools/latent"
    DESCRIPTION = (
        "Stages a generated transition between two AV clips. Wire prior_av_latent "
        "-> optional_prior_av_latent, bridge_latent -> latents, end_image -> "
        "optional_cond_images, cond_indices -> optional_cond_image_indices. The "
        "head is pinned by real AV (motion and audio), the tail by B's first frame."
    )

    def prep(self, clip_a, clip_b, vae, bridge_seconds, a_tail_seconds, fps):
        label = "LTXAVBridgePrep"
        va, aa = _split_av(clip_a, label)
        vb, ab = _split_av(clip_b, label)
        _check_compatible(va, vb, 2, label)

        # --- prior: clip A, optionally just its tail -------------------------
        T_a_v = va.shape[2]
        if a_tail_seconds <= 0:
            k = T_a_v
        else:
            k = max(1, min(T_a_v, round(a_tail_seconds * fps / 8.0)))

        if k < T_a_v:
            # Slice aligned at the END, because that is where the continuation
            # happens. A mid-sequence latent owns 8 audio frames but a standalone
            # clip's latent 0 owns only 1, so any slice carries a (q - c) frame
            # surplus; taking the LAST audio_pos(k) frames puts that discrepancy at
            # the prior's head, which is discarded context, instead of at the seam.
            prior_v = va[:, :, -k:].contiguous()
            prior_a = aa[:, :, -audio_pos(k, fps):].contiguous()
        else:
            prior_v, prior_a = va, aa

        # --- bridge: empty AV of the requested length ------------------------
        bridge_px = max(8, int(round(bridge_seconds * fps / 8.0)) * 8)
        T_c = bridge_px // 8
        T_c_a = audio_pos(T_c, fps)
        B, C_v, _, H, W = va.shape
        C_a, F_s = aa.shape[1], aa.shape[3]
        dev, dty = va.device, va.dtype
        bridge = {"samples": NestedTensor([
            torch.zeros(B, C_v, T_c, H, W, device=dev, dtype=dty),
            torch.zeros(B, C_a, T_c_a, F_s, device=dev, dtype=dty),
        ])}

        # --- tail pin: clip B's first frame ----------------------------------
        # B's latent 0 IS the video-start latent, so it decodes to exactly one
        # pixel frame — B's first frame, no trimming needed.
        end_image = vae.decode(vb[:, :, :1])
        if end_image.ndim == 5:               # [B,T,H,W,C] -> [T,H,W,C]
            end_image = end_image.reshape(-1, *end_image.shape[-3:])
        end_image = end_image[:1]

        info = (
            f"prior: {prior_v.shape[2]} video / {prior_a.shape[2]} audio latents "
            f"({'all of A' if k >= T_a_v else f'{k} latent tail, {k * 8 / fps:.2f}s'}) | "
            f"bridge: {T_c} video / {T_c_a} audio latents "
            f"({bridge_px} px, {bridge_px / fps:.2f}s @ {fps:g}fps) | "
            f"tail pinned to clip B frame 0 at index -1.\n"
            f"Reassemble: sampler output is prior+bridge. To isolate the bridge, "
            f"feed trim_prior_latents ({prior_v.shape[2] - 1}) to LTX AV Trim "
            f"Latents as drop_head/latents — that deliberately KEEPS the prior's "
            f"last latent so Join has a stub to drop, leaving the bridge whole."
        )
        print(f"[{label}] {info}")

        return (
            {"samples": NestedTensor([prior_v, prior_a])},
            bridge,
            end_image,
            "-1",
            info,
            # Slices have no stub latent, but Join drops one from every clip after
            # the first — so trimming the prior off exactly would eat a latent of
            # bridge. Leaving the prior's final latent in place gives Join the stub
            # it expects and the bridge survives intact.
            max(0, prior_v.shape[2] - 1),
        )


class LTXAVSourceMatch:
    """Probe two source videos and emit the settings that make them match.

    Everything here exists so normalisation happens at the LOADERS, in pixel
    space, before anything is encoded — where VHS does real frame resampling and
    real image resizing. Reconciling later, in the latent domain, is possible
    (Join will do it) but it is an approximation: audio gets stretched and
    latents get interpolated. Probing first makes that path never run.

    One node decides rate AND size for the whole graph, from the files
    themselves, with no math nodes and nothing typed twice.
    """

    @classmethod
    def INPUT_TYPES(s):
        from .video_cut_marker import _list_input_videos
        vids = _list_input_videos()
        return {
            "required": {
                "video_a": (vids, {"tooltip":
                    "First source. Convert to an input to wire a Cut Marker's "
                    "video_path instead of picking here."}),
                "video_b": (vids, {"tooltip": "Second source."}),
            },
            "optional": {
                "dim_multiple": ("INT", {"default": 32, "min": 8, "max": 128, "step": 8,
                    "tooltip": "Round the emitted size DOWN to this multiple. LTX "
                               "needs 32; use 64 if a later stage halves the "
                               "resolution and must stay valid."}),
                "snap_fps_to_int": ("BOOLEAN", {"default": True, "tooltip":
                    "Round 23.976 -> 24, 29.97 -> 30. Containers report broadcast "
                    "rates as fractions; the integer keeps every downstream widget "
                    "on the same number."}),
                "fallback_fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 200.0,
                    "step": 0.01, "tooltip":
                    "Used for a file whose rate cannot be read (and reported)."}),
            },
        }

    # APPEND-ONLY: half dims last. ComfyUI links outputs by index.
    RETURN_TYPES = ("FLOAT", "INT", "INT", "INT", "STRING", "STRING", "STRING",
                    "INT", "INT")
    RETURN_NAMES = ("fps", "fps_int", "width", "height", "path_a", "path_b",
                    "info", "width_half", "height_half")
    FUNCTION = "match"
    CATEGORY = "LTXAVTools/utils"
    DESCRIPTION = (
        "Reads both sources' fps and dimensions (metadata only, ~15 ms, no decode) "
        "and emits the LOWEST rate and SMALLEST size, snapped to a valid LTX "
        "multiple, plus their paths. Drive both loaders' force_rate / custom_width "
        "/ custom_height / video from this, and the rest of the graph's rate "
        "widgets from fps — then nothing downstream has to reconcile anything."
    )

    @classmethod
    def IS_CHANGED(s, video_a, video_b, **kw):
        return f"{video_a}|{video_b}"

    @staticmethod
    def _resolve(x):
        """Accept an input-dir filename (combo) or a full path (wired)."""
        import os
        x = (x or "").strip()
        if os.path.isfile(x):
            return x
        try:
            import folder_paths
            r = folder_paths.get_annotated_filepath(x)
            if r and os.path.isfile(r):
                return r
        except Exception:
            pass
        return x

    def _probe(self, path, snap, fallback, lines):
        """-> (fps, width, height); width/height are 0 if unreadable."""
        import os
        f, w, h = None, 0, 0
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                v = cap.get(cv2.CAP_PROP_FPS)
                if v and v > 0:
                    f = float(v)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()
        except Exception as e:
            lines.append(f"  {os.path.basename(path)}: probe failed ({e})")
        name = os.path.basename(path)
        if f is None:
            f = float(fallback)
            lines.append(f"  {name}: fps UNREADABLE -> fallback {f:g}, {w}x{h}")
            return f, w, h
        shown = float(round(f)) if (snap and abs(f - round(f)) < 0.05) else f
        lines.append(f"  {name}: {f:.3f} fps"
                     + (f" -> {shown:g}" if shown != f else "")
                     + f", {w}x{h}")
        return shown, w, h

    def match(self, video_a, video_b, dim_multiple=32, snap_fps_to_int=True,
              fallback_fps=25.0):
        label = "LTXAVSourceMatch"
        pa, pb = self._resolve(video_a), self._resolve(video_b)
        lines = []
        fa, wa, ha = self._probe(pa, snap_fps_to_int, fallback_fps, lines)
        fb, wb, hb = self._probe(pb, snap_fps_to_int, fallback_fps, lines)

        fps = min(fa, fb)

        m = max(8, int(dim_multiple))
        dims = [(w, h) for w, h in ((wa, ha), (wb, hb)) if w > 0 and h > 0]
        if not dims:
            raise ValueError(
                f"[{label}] could not read dimensions from either source. Check the "
                f"files are readable video."
            )
        # Smallest by AREA, then snapped DOWN so the size is always achievable and
        # never upscales a source.
        w0, h0 = min(dims, key=lambda d: d[0] * d[1])
        width, height = (w0 // m) * m, (h0 // m) * m
        if width < m or height < m:
            raise ValueError(
                f"[{label}] source is {w0}x{h0}, which rounds to {width}x{height} at "
                f"multiple {m}. Lower dim_multiple or use a larger source."
            )

        if len(dims) == 2:
            ar = [w / h for w, h in dims]
            if max(ar) / min(ar) > 1.02:
                lines.append(
                    f"  NOTE: aspects differ ({ar[0]:.3f}:1 vs {ar[1]:.3f}:1). With "
                    f"BOTH custom_width and custom_height set, VHS CROPS to fit — "
                    f"framing will change on the mismatched clip.")

        # Two-stage pipelines generate at HALF and latent-upscale x2 back, so the
        # half must itself be a valid LTX grid. That needs the full dims divisible
        # by 2*32 = 64 — which is what dim_multiple 64 buys you.
        wh, hh = width // 2, height // 2
        if wh % 32 or hh % 32:
            lines.append(
                f"  NOTE: half is {wh}x{hh}, not a multiple of 32 — a x2 latent "
                f"upscale will not land on {width}x{height}. Set dim_multiple = 64 "
                f"if you generate at half and upscale back.")

        info = chr(10).join(
            [f"match: {fps:g} fps, {width}x{height} (from {w0}x{h0}, snapped "
             f"to /{m}) | half {wh}x{hh}"] + lines)
        print(f"[{label}] {info}")
        return (fps, int(round(fps)), width, height, pa, pb, info, wh, hh)


def _q_c(fps):
    """(q, c) — audio frames owned by a mid-sequence latent, and by latent 0."""
    return audio_pos(2, fps) - audio_pos(1, fps), audio_pos(1, fps)


class LTXAVBridgeCompose:
    """Build [A_tail | free bridge | B_head] as one latent, plus its denoise mask.

    Both boundary conditions are REAL AV — motion and audio on each side — so the
    bridge is generated between two known states rather than converging on a
    still. The sampler regenerates only the masked span; the tails are frozen
    context.

    Audio allocation is the fiddly part and is why this is a node. Each section
    must own exactly the audio its position in the COMPOSED clip implies:

        A_tail : audio_pos(k_a)  — end-aligned, so the bridge seam is exact and
                                   the first-frame discrepancy lands at the
                                   composed clip's head (frozen context)
        bridge : k_c * q         — exactly the free span the mask will describe
        B_head : k_b * q         — start-aligned at the seam, so that seam is
                                   exact too and the discrepancy lands at the tail

    Total = audio_pos(k_a) + (k_c + k_b) * q = audio_pos(k_a + k_c + k_b). Both
    seams exact; both discrepancies pushed to the outer edges, away from the
    bridge.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_a": ("LATENT", {"tooltip": "Outgoing clip (full)."}),
                "clip_b": ("LATENT", {"tooltip": "Incoming clip (full)."}),
                "a_tail_seconds": ("FLOAT", {
                    "default": 2.0, "min": 0.08, "max": 30.0, "step": 0.04,
                    "tooltip": "How much of A's ending to freeze as leading context.",
                }),
                "bridge_seconds": ("FLOAT", {
                    "default": 1.0, "min": 0.08, "max": 20.0, "step": 0.04,
                    "tooltip": "The span to generate. Short is easier — it has to "
                               "arrive at B's actual state, and every extra second "
                               "is more room to drift first.",
                }),
                "b_head_seconds": ("FLOAT", {
                    "default": 2.0, "min": 0.08, "max": 30.0, "step": 0.04,
                    "tooltip": "How much of B's opening to freeze as trailing context.",
                }),
                "fps": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 200.0, "step": 0.01,
                    "tooltip": "0 = AUTO: read each clip's rate from its own "
                               "video:audio ratio, lowest wins. The resolved rate is "
                               "in `info` and the plan — set the sampler's video_fps "
                               "and that pass's LTXVConditioning frame_rate to it.",
                }),
                "feather": ("INT", {
                    "default": 8, "min": 0, "max": 64,
                    "tooltip": "Pixel frames of ramp on each side of the free span, "
                               "so the seams are not hard mask edges.",
                }),
                "on_size_mismatch": (["downscale_to_smallest", "error"], {
                    "default": "downscale_to_smallest",
                    "tooltip": "Clips of different RESOLUTION but matching aspect "
                               "reconcile automatically to the smallest. Latent-grid "
                               "resampling is lossy — pixel-space resize + re-encode "
                               "keeps more detail. Set error to refuse instead. "
                               "Differing ASPECT always errors: cropping or "
                               "letterboxing changes framing, so that is your call.",
                }),
                "bridge_latent": ("LATENT", {
                    "tooltip": "STAGE 2. A bridge that already exists — normally "
                               "Bridge Extract's bridge_only, latent-upscaled. The "
                               "free span is seeded with it instead of zeros, so a "
                               "LIGHT denoise refines it rather than regenerating "
                               "from nothing. Latent upsampling alone does not look "
                               "right; it needs this pass. Leave unconnected for "
                               "stage 1.",
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "MASK", "LTXAV_BRIDGE_PLAN", "STRING")
    RETURN_NAMES = ("latent", "denoise_mask", "plan", "info")
    FUNCTION = "compose"
    CATEGORY = "LTXAVTools/latent"
    DESCRIPTION = (
        "Composes [A_tail | bridge | B_head] with a matching denoise mask. Wire "
        "latent -> sampler .latents and denoise_mask -> .optional_denoise_mask "
        "(the sampler derives the AUDIO freeze from it automatically). Pass plan "
        "to LTX AV Bridge Extract to rebuild at full resolution."
    )

    def compose(self, clip_a, clip_b, a_tail_seconds, bridge_seconds,
                b_head_seconds, fps, feather,
                on_size_mismatch="downscale_to_smallest", bridge_latent=None):
        label = "LTXAVBridgeCompose"
        va, aa = _split_av(clip_a, label)
        vb, ab = _split_av(clip_b, label)
        fps, (pa, pb) = _resolve_fps([(va, aa), (vb, ab)], fps, label)
        (va, aa), (vb, ab) = pa, pb
        va, vb = _reconcile_sizes([va, vb], on_size_mismatch, label)
        q, c = _q_c(fps)

        def n_lat(sec):
            return max(1, int(round(sec * fps / 8.0)))

        k_a = min(va.shape[2], n_lat(a_tail_seconds))
        k_b = min(vb.shape[2], n_lat(b_head_seconds))
        k_c = n_lat(bridge_seconds)
        T = k_a + k_c + k_b

        # Section audio counts are DIFFERENCES of the global boundary map, never
        # k * q — q = 200/fps is only an integer on the exact-rate set, so a
        # multiply silently breaks at 24 fps. Differences telescope to
        # audio_pos(T) at ANY rate (same principle as SPEC_50FPS).
        a_take = audio_pos(k_a, fps)
        c_take = audio_pos(k_a + k_c, fps) - a_take
        b_take = audio_pos(T, fps) - audio_pos(k_a + k_c, fps)
        if a_take > aa.shape[2] or b_take > ab.shape[2]:
            raise ValueError(
                f"[{label}] need {a_take} audio latents from A (has {aa.shape[2]}) "
                f"and {b_take} from B (has {ab.shape[2]}). Shorten the tails."
            )

        B_, C_v, _, H, W = va.shape
        C_a, F_s = aa.shape[1], aa.shape[3]
        dev, dty = va.device, va.dtype

        if bridge_latent is None:
            mid_v = torch.zeros(B_, C_v, k_c, H, W, device=dev, dtype=dty)
            mid_a = torch.zeros(B_, C_a, c_take, F_s, device=dev, dtype=dty)
            seeded = "zeros (stage 1)"
        else:
            bv, ba = _split_av(bridge_latent, label)
            if bv.shape[2] != k_c or ba.shape[2] != c_take:
                raise ValueError(
                    f"[{label}] bridge_latent is {bv.shape[2]} video / {ba.shape[2]} "
                    f"audio latents but this composite needs {k_c} / {c_take}. It "
                    f"must come from the SAME seconds/fps settings — a stage-2 "
                    f"compose has to describe the same timeline as stage 1."
                )
            if (bv.shape[3], bv.shape[4]) != (H, W):
                raise ValueError(
                    f"[{label}] bridge_latent is {bv.shape[4]}x{bv.shape[3]} but the "
                    f"clips are {W}x{H}. Latent-upscale the bridge to full "
                    f"resolution before feeding it back."
                )
            mid_v = bv.to(device=dev, dtype=dty)
            mid_a = ba.to(device=dev, dtype=dty)
            seeded = "pre-formed bridge (stage 2 — use a LIGHT denoise)"

        video = torch.cat([va[:, :, -k_a:], mid_v, vb[:, :, :k_b]], dim=2)
        audio = torch.cat([aa[:, :, -a_take:], mid_a, ab[:, :, :b_take]], dim=2)

        want_a = audio_pos(T, fps)
        if audio.shape[2] != want_a:
            raise AssertionError(
                f"[{label}] internal: audio {audio.shape[2]} != audio_pos({T}, "
                f"{fps:g}) = {want_a} (sections {a_take}+{c_take}+{b_take})")

        px = (T - 1) * 8 + 1
        s_px = 8 * k_a - 7
        e_px = min(px, 8 * (k_a + k_c) - 7)
        prof = torch.zeros(px, dtype=torch.float32)
        prof[s_px:e_px] = 1.0
        if feather > 0:
            f = min(feather, max(1, (e_px - s_px) // 2))
            ramp = torch.linspace(0, 1, f + 2, dtype=torch.float32)[1:-1]
            prof[s_px:s_px + f] = ramp
            prof[e_px - f:e_px] = ramp.flip(0)
        mask = prof.view(px, 1, 1).expand(px, 64, 64).contiguous()

        plan = {"k_a": k_a, "k_c": k_c, "k_b": k_b, "fps": float(fps), "T": T}
        info = (f"[{seeded}] A_tail {k_a} + bridge {k_c} + B_head {k_b} = {T} video latents "
                f"({px} px, {px / fps:.2f}s @ {fps:g}fps), {audio.shape[2]} audio. "
                f"Free span: video [{k_a},{k_a + k_c}) = px [{s_px},{e_px}), "
                f"audio [{audio_pos(k_a, fps)},{audio_pos(k_a + k_c, fps)}).")
        print(f"[{label}] {info}")
        return ({"samples": NestedTensor([video, audio])}, mask, plan, info)


class LTXAVBridgeExtract:
    """Pull the generated bridge out of a sampled composite and rebuild at full res."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sampled": ("LATENT", {
                    "tooltip": "The sampler's output for the composed latent "
                               "(optionally spatially upscaled — latent COUNTS must "
                               "be unchanged, so no temporal upsampling).",
                }),
                "plan": ("LTXAV_BRIDGE_PLAN",),
                "clip_a": ("LATENT", {"tooltip": "Full-resolution clip A (original)."}),
                "clip_b": ("LATENT", {"tooltip": "Full-resolution clip B (original)."}),
            },
            "optional": {
                "mask_feather": ("INT", {
                    "default": 8, "min": 0, "max": 64,
                    "tooltip": "Feather for the emitted region_mask, if you refine "
                               "the bridge again at full resolution.",
                }),
            },
        }

    # APPEND-ONLY: bridge_only last. ComfyUI links outputs by index.
    RETURN_TYPES = ("LATENT", "MASK", "STRING", "LATENT")
    RETURN_NAMES = ("latent", "region_mask", "info", "bridge_only")
    FUNCTION = "extract"
    CATEGORY = "LTXAVTools/latent"
    DESCRIPTION = (
        "Discards the scaffolding tails, keeps only the generated bridge, and "
        "rejoins it between the ORIGINAL full-resolution clips. region_mask frees "
        "just the bridge for an optional full-res refinement pass."
    )

    def extract(self, sampled, plan, clip_a, clip_b, mask_feather=8):
        label = "LTXAVBridgeExtract"
        k_a, k_c, k_b = plan["k_a"], plan["k_c"], plan["k_b"]
        fps, T_expect = plan["fps"], plan["T"]
        q, c = _q_c(fps)

        sv, sa = _split_av(sampled, label)
        if sv.shape[2] != T_expect:
            raise ValueError(
                f"[{label}] sampled latent has {sv.shape[2]} video latents but the "
                f"plan describes {T_expect}. Latent COUNTS must survive the upscale "
                f"— a spatial upsampler preserves them, a TEMPORAL one does not "
                f"(L -> 2L-1). Re-compose if the timeline changed."
            )

        # Take one extra latent at the head — A_tail's last — so Join has a stub to
        # drop. Slices carry no stub of their own, and Join drops one from every
        # clip after the first, so an exact cut would eat a latent of bridge.
        s_lat = max(0, k_a - 1)
        piece_v = sv[:, :, s_lat:k_a + k_c]
        n_piece = piece_v.shape[2]
        # End-aligned audio: the bridge's END is the seam with B, so keep that
        # exact and let the discrepancy fall on the stub Join discards.
        a_end = audio_pos(k_a + k_c, fps)
        a_take = audio_pos(n_piece, fps)
        piece_a = sa[:, :, max(0, a_end - a_take):a_end]

        piece = {"samples": NestedTensor([piece_v.contiguous(), piece_a.contiguous()])}
        # fps=0 so Join re-resolves: the ORIGINALS may be at different rates than
        # the plan (the plan rate is min(A, B), and the extracted piece already
        # carries it), so Join picks that same rate and retimes whichever
        # original is faster. Passing the plan rate directly would reject them.
        joined, _, jinfo = LTXAVJoinLatents().join(clip_a, piece, 0.0,
                                                   latent_3=clip_b)

        jv = joined["samples"].tensors[0]
        T_a_lat = _split_av(clip_a, label)[0].shape[2]
        px = (jv.shape[2] - 1) * 8 + 1
        spans = [(0, T_a_lat), (T_a_lat, T_a_lat + k_c)]
        mask = LTXAVJoinLatents._region_mask(spans, jv.shape[2], px, "2",
                                             mask_feather, label)

        # The bridge alone, at its exact owned span — no stub. This is what a
        # stage-2 compose seeds its free region with, so latent upsampling gets
        # the light denoise pass it needs to actually look right.
        bridge_only = {"samples": NestedTensor([
            sv[:, :, k_a:k_a + k_c].contiguous(),
            sa[:, :, audio_pos(k_a, fps):a_end].contiguous(),
        ])}

        info = (f"bridge {k_c} latents extracted (piece {n_piece} incl. stub) -> "
                f"rejoined with full-res A ({T_a_lat}) and B: {jv.shape[2]} video "
                f"latents ({px} px, {px / fps:.2f}s @ {fps:g}fps). {jinfo}")
        print(f"[{label}] {info}")
        return (joined, mask, info, bridge_only)


class LTXAVTrimLatents:
    """Keep or drop a span from either end of an AV latent, audio included."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "mode": (["keep_head", "keep_tail", "drop_head", "drop_tail"], {
                    "default": "keep_tail",
                    "tooltip": "keep_head: clip B's opening. keep_tail: clip A's "
                               "closing. drop_head: strip a bridge pass's prior span "
                               "so only the generated bridge remains. drop_tail: the "
                               "mirror.",
                }),
                "amount": ("FLOAT", {
                    "default": 2.0, "min": 0.04, "max": 600.0, "step": 0.04,
                    "tooltip": "How much to keep or drop, in the chosen units.",
                }),
                "units": (["seconds", "latents"], {
                    "default": "seconds",
                    "tooltip": "latents is exact — use it when undoing a known "
                               "latent count, e.g. a bridge pass's prior span.",
                }),
                "fps": ("FLOAT", {
                    "default": 25.0, "min": 1.0, "max": 200.0, "step": 0.01,
                    "tooltip": "Must match the rate the clip was generated at.",
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "STRING")
    RETURN_NAMES = ("latent", "latent_count", "info")
    FUNCTION = "trim"
    CATEGORY = "LTXAVTools/latent"
    DESCRIPTION = (
        "Keeps or drops a span from either end of an AV latent, slicing the audio "
        "to match via audio_pos. Head slices are exact; tail slices are aligned at "
        "their END, which is what matters for continuation."
    )

    def trim(self, latent, mode, amount, units, fps):
        label = "LTXAVTrimLatents"
        v, a = _split_av(latent, label)
        _clip_fps_check(v, a, fps, label, "input")
        T = v.shape[2]

        n = (max(1, int(round(amount))) if units == "latents"
             else max(1, int(round(amount * fps / 8.0))))

        keep_head = mode == "keep_head" or mode == "drop_tail"
        k = n if mode.startswith("keep") else T - n
        if k < 1 or k > T:
            raise ValueError(
                f"[{label}] {mode} {n} on a {T}-latent clip leaves {k} latents. "
                f"Amount must be smaller than the clip "
                f"({T} latents = {(T - 1) * 8 + 1} px = {((T - 1) * 8 + 1) / fps:.2f}s "
                f"@ {fps:g}fps)."
            )

        want_a = audio_pos(k, fps)
        if keep_head:
            # A head slice keeps latent 0 — the genuine video-start latent — so it
            # is a well-formed standalone clip and the audio slice is exact.
            v2, a2 = v[:, :, :k], a[:, :, :want_a]
        else:
            # A tail slice is aligned at its END, because that is where a join or a
            # continuation happens. A mid-sequence latent owns q audio frames but a
            # standalone clip's latent 0 owns only c, so any tail carries a (q - c)
            # surplus; taking the LAST audio_pos(k) frames puts that at the slice's
            # head instead of at the seam.
            v2, a2 = v[:, :, -k:], a[:, :, -want_a:]

        if a2.shape[2] != want_a:
            print(f"[{label}] audio slice is {a2.shape[2]}, wanted {want_a} — the "
                  f"input's audio ({a.shape[2]}) did not match its video ({T} "
                  f"latents, expects {audio_pos(T, fps)}).")

        px = (k - 1) * 8 + 1
        info = (f"{mode} {n} {units} -> {k}/{T} video latents ({px} px, "
                f"{px / fps:.2f}s @ {fps:g}fps), {a2.shape[2]} audio latents.")
        print(f"[{label}] {info}")
        return ({"samples": NestedTensor([v2.contiguous(), a2.contiguous()])}, k, info)


NODE_CLASS_MAPPINGS = {
    "LTXAVSourceMatch": LTXAVSourceMatch,
    "LTXAVBridgeCompose": LTXAVBridgeCompose,
    "LTXAVBridgeExtract": LTXAVBridgeExtract,
    "LTXAVJoinLatents": LTXAVJoinLatents,
    "LTXAVBridgePrep": LTXAVBridgePrep,
    "LTXAVTrimLatents": LTXAVTrimLatents,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXAVSourceMatch": "LTX AV Source Match",
    "LTXAVBridgeCompose": "LTX AV Bridge Compose",
    "LTXAVBridgeExtract": "LTX AV Bridge Extract",
    "LTXAVJoinLatents": "LTX AV Join Latents",
    "LTXAVBridgePrep": "LTX AV Bridge Prep",
    "LTXAVTrimLatents": "LTX AV Trim Latents",
}
