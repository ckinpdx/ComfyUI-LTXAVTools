from math import gcd

try:
    from server import PromptServer
    from aiohttp import web
    _HAS_SERVER = True
except ImportError:
    _HAS_SERVER = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRID           = 64
TOLERANCE      = 0.03
LANDSCAPE_MAX  = 3776
PORTRAIT_MAX_H = 1920
PORTRAIT_MAX_W = 1088
SHORT_MIN      = 512

_STAGE_GRID = {1: 32, 2: 64, 3: 128}


def _snap(x, grid: int) -> int:
    """Snap x to the nearest positive multiple of grid (LTX-valid resolution).
    Returns 0 for non-positive/None input (used as the bypass sentinel)."""
    if not x or x <= 0:
        return 0
    return max(grid, int(round(x / grid)) * grid)

# ---------------------------------------------------------------------------
# Ratio definitions
# ---------------------------------------------------------------------------
RATIOS = [
    (16,  9, "16:9 — YouTube, HD, TV",      "9:16 — TikTok, Reels, Shorts"),
    (21,  9, "21:9 — Ultrawide, cinematic", "9:21 — Ultrawide portrait"),
    ( 4,  3, "4:3 — Classic TV, monitor",   "3:4 — Tablet portrait"),
    ( 3,  2, "3:2 — Photography, DSLR",     "2:3 — Portrait photo"),
    ( 2,  1, "2:1 — Cinematic wide",        "1:2 — Tall mobile"),
    ( 5,  4, "5:4 — Old CRT monitor",       "4:5 — Instagram portrait"),
    ( 1,  1, "1:1 — Square, Instagram",     "1:1 — Square, Instagram"),
]

# ---------------------------------------------------------------------------
# Core calculation
# ---------------------------------------------------------------------------
def _build_options(ratio_long: int, ratio_short: int, landscape: bool, grid: int = GRID) -> list:
    target    = ratio_long / max(ratio_short, 1)
    max_long  = ((LANDSCAPE_MAX  if landscape else PORTRAIT_MAX_H) // grid) * grid
    max_short = ((LANDSCAPE_MAX  if landscape else PORTRAIT_MAX_W) // grid) * grid
    short_min = ((SHORT_MIN + grid - 1) // grid) * grid
    max_a     = max_long // grid

    candidates = {}

    for a in range(1, max_a + 1):
        long_px = a * grid
        for b in range(1, a + 1):
            short_px = b * grid
            if short_px > max_short:
                break
            if short_px < short_min:
                continue
            dev = abs((long_px / short_px) - target) / target
            if dev <= TOLERANCE:
                candidates[(long_px, short_px)] = dev

    if not candidates:
        return []

    by_long = {}
    for (l, s), dev in candidates.items():
        if l not in by_long or dev < by_long[l][1]:
            by_long[l] = ((l, s), dev)

    by_short = {}
    for (l, s), dev in by_long.values():
        if s not in by_short or dev < by_short[s][1]:
            by_short[s] = ((l, s), dev)

    ordered = sorted(by_short.values(), key=lambda x: x[0][0] * x[0][1])
    result = [f"{l}x{s}" if landscape else f"{s}x{l}" for (l, s), _ in ordered]

    if ratio_long == ratio_short and grid == GRID and _SQUARE_CAP is not None and len(result) > _SQUARE_CAP:
        result = _trim_evenly(result, _SQUARE_CAP)

    return result


def _trim_evenly(lst: list, n: int) -> list:
    indices = sorted({round(i * (len(lst) - 1) / (n - 1)) for i in range(n)})
    return [lst[i] for i in indices]


_SQUARE_CAP = None
_SQUARE_CAP = max(
    (len(_build_options(rl, rs, land))
     for rl, rs, *_ in RATIOS if rl != rs
     for land in (True, False)),
    default=15,
)

_DEFAULT_OPTS = _build_options(16, 9, landscape=True)


# ---------------------------------------------------------------------------
# LTXDimensionCalculator
# ---------------------------------------------------------------------------
class LTXDimensionCalculator:
    CATEGORY = "LTXAVTools"

    @classmethod
    def INPUT_TYPES(cls):
        ratio_labels = [ls for _, _, ls, _ in RATIOS]
        mid = len(_DEFAULT_OPTS) // 2
        return {
            "required": {
                "ratio":        (ratio_labels, {
                    "default": ratio_labels[0],
                    "tooltip": "Common aspect ratios and their typical applications.",
                }),
                "orientation":  (["Landscape", "Portrait"], {
                    "default": "Landscape",
                    "tooltip": "Switches between landscape and portrait resolution lists.",
                }),
                "resolution":   (_DEFAULT_OPTS, {
                    "default": _DEFAULT_OPTS[mid],
                    "tooltip": "All options are divisible by 64 (LTX-compatible).",
                }),
            },
            "optional": {
                "use_custom": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Override the dropdown with custom_width/custom_height. "
                               "Toggle-controlled (not value-based), so a bypassed upstream "
                               "node can't accidentally switch modes.",
                }),
                "custom_role": (["half (stage 1)", "full (final)"], {
                    "default": "half (stage 1)",
                    "tooltip": "Is the custom size your stage-1 (half) resolution or the "
                               "final (full) resolution? 'half (stage 1)' sets full = 2x "
                               "custom (input-video case: stage 1 matches the source, "
                               "stage 2 doubles). 'full (final)' sets half = custom / 2.",
                }),
                "custom_width": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "Custom width — used only when use_custom is on. Snapped to "
                               "the valid grid (÷32 in half role, ÷64 in full role).",
                }),
                "custom_height": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "Custom height — used only when use_custom is on.",
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    RETURN_TYPES  = ("INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES  = ("width", "height", "width_half", "height_half", "label")
    FUNCTION      = "calculate"

    def calculate(self, ratio: str, orientation: str, resolution: str,
                  use_custom: bool = False, custom_role: str = "half (stage 1)",
                  custom_width: int = 0, custom_height: int = 0):
        if use_custom:
            cw = _snap(custom_width, 32 if str(custom_role).startswith("half") else 64)
            ch = _snap(custom_height, 32 if str(custom_role).startswith("half") else 64)
            if cw > 0 and ch > 0:
                if str(custom_role).startswith("half"):
                    # custom = stage-1 (half) res; full = 2x → stage 2 doubles
                    fw, fh = cw * 2, ch * 2
                    return (fw, fh, cw, ch, f"{cw}x{ch} (stage1) -> {fw}x{fh} (final)")
                # custom = full (final) res; half = /2 (÷64 keeps it ÷32)
                hw, hh = cw // 2, ch // 2
                return (cw, ch, hw, hh, f"{cw}x{ch} (final) -> {hw}x{hh} (stage1)")
            print("[LTXDimensionCalculator] use_custom on but custom dims <= 0 "
                  "(upstream bypassed?); falling back to the dropdown.")
        w, h = map(int, resolution.split("x"))
        return (w, h, w // 2, h // 2, resolution)


# ---------------------------------------------------------------------------
# LTXDimensionCalculator3Stage
# ---------------------------------------------------------------------------
_DEFAULT_OPTS_3STAGE = _build_options(16, 9, landscape=True, grid=128)

class LTXDimensionCalculator3Stage:
    CATEGORY = "LTXAVTools"

    @classmethod
    def INPUT_TYPES(cls):
        ratio_labels = [ls for _, _, ls, _ in RATIOS]
        mid = len(_DEFAULT_OPTS_3STAGE) // 2
        default = _DEFAULT_OPTS_3STAGE[mid] if _DEFAULT_OPTS_3STAGE else "1024x576"
        return {
            "required": {
                "ratio":       (ratio_labels, {
                    "default": ratio_labels[0],
                    "tooltip": "Common aspect ratios and their typical applications.",
                }),
                "orientation": (["Landscape", "Portrait"], {
                    "default": "Landscape",
                    "tooltip": "Switches between landscape and portrait resolution lists.",
                }),
                "resolution":  (_DEFAULT_OPTS_3STAGE or [default], {
                    "default": default,
                    "tooltip": "All options are divisible by 128.",
                }),
            },
            "optional": {
                "use_custom": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Override the dropdown with custom_width/custom_height. "
                               "Toggle-controlled (not value-based), so a bypassed upstream "
                               "node can't accidentally switch modes.",
                }),
                "custom_role": (["quarter (stage 1)", "half (stage 2)", "full (final)"], {
                    "default": "quarter (stage 1)",
                    "tooltip": "Which stage the custom size describes. 'quarter (stage 1)' "
                               "sets full = 4x custom (input-video case: stage 1 matches "
                               "the source). 'half (stage 2)' sets full = 2x custom. "
                               "'full (final)' derives the halves down from custom.",
                }),
                "custom_width": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "Custom width — used only when use_custom is on. Snapped to "
                               "the role's grid (÷32 quarter, ÷64 half, ÷128 full) so every "
                               "derived stage stays LTX-valid.",
                }),
                "custom_height": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "Custom height — used only when use_custom is on.",
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    RETURN_TYPES  = ("INT", "INT", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES  = ("width", "height", "width_half", "height_half", "width_quarter", "height_quarter", "label")
    FUNCTION      = "calculate"

    def calculate(self, ratio: str, orientation: str, resolution: str,
                  use_custom: bool = False, custom_role: str = "quarter (stage 1)",
                  custom_width: int = 0, custom_height: int = 0):
        if use_custom:
            role = str(custom_role)
            grid = 32 if role.startswith("quarter") else (64 if role.startswith("half") else 128)
            cw = _snap(custom_width, grid)
            ch = _snap(custom_height, grid)
            if cw > 0 and ch > 0:
                mult = 4 if role.startswith("quarter") else (2 if role.startswith("half") else 1)
                fw, fh = cw * mult, ch * mult
                label = f"{fw // 4}x{fh // 4} -> {fw // 2}x{fh // 2} -> {fw}x{fh}"
                return (fw, fh, fw // 2, fh // 2, fw // 4, fh // 4, label)
            print("[LTXDimensionCalculator3Stage] use_custom on but custom dims <= 0 "
                  "(upstream bypassed?); falling back to the dropdown.")
        w, h = map(int, resolution.split("x"))
        return (w, h, w // 2, h // 2, w // 4, h // 4, resolution)


# ---------------------------------------------------------------------------
# LTXFrameCalculator
# ---------------------------------------------------------------------------
class LTXFrameCalculator:
    CATEGORY = "LTXAVTools"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seconds": ("FLOAT", {
                    "default": 5.0, "min": 0.1, "step": 0.1,
                    "tooltip": "Desired clip duration in seconds.",
                }),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0,
                    "tooltip": "Frames per second.",
                }),
            }
        }

    RETURN_TYPES  = ("INT", "INT", "FLOAT")
    RETURN_NAMES  = ("frame_count", "latent_frames", "actual_seconds")
    FUNCTION      = "calculate"

    def calculate(self, seconds: float, fps: float):
        raw = seconds * fps
        n = max(1, round((raw - 1) / 8))
        frames = 8 * n + 1
        latent = n + 1
        actual = frames / fps
        return (frames, latent, round(actual, 4))


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------
if _HAS_SERVER:
    @PromptServer.instance.routes.get("/ltx-dim-calc/resolutions")
    async def _get_resolutions(request):
        ratio_str   = request.rel_url.query.get("ratio", "16:9")
        orientation = request.rel_url.query.get("orientation", "Landscape")
        stages      = int(request.rel_url.query.get("stages", "2"))
        landscape   = orientation == "Landscape"
        grid        = _STAGE_GRID.get(stages, GRID)

        try:
            rw, rh = map(int, ratio_str.split(":"))
        except (ValueError, IndexError):
            return web.json_response([])

        g  = gcd(rw, rh)
        rw, rh = rw // g, rh // g
        ratio_long  = max(rw, rh)
        ratio_short = min(rw, rh)

        return web.json_response(_build_options(ratio_long, ratio_short, landscape, grid=grid))


class LTXSceneLengthCalculator:
    """
    Converts a pipe/comma-separated list of scene durations (seconds) into the
    scene_lengths string for the AV Looping Sampler plus the exactly-matching
    total frame_count for the empty latent. Single source of truth: the sampler
    schedule and the latent length cannot disagree.

    Each scene snaps to latent granularity (multiples of 8 pixel frames).
    Total frame_count = (sum_of_scene_latents - 1) * 8 + 1.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "scene_seconds": ("STRING", {
                    "multiline": False,
                    "default": "10 | 10 | 10",
                    "tooltip": "Scene durations in seconds, separated by '|' or ','. "
                               "One scene per chunk / prompt segment.",
                }),
                "fps": ("FLOAT", {
                    "default": 25.0, "min": 1.0, "max": 120.0, "step": 0.01,
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "STRING", "FLOAT")
    RETURN_NAMES = ("scene_lengths", "frame_count", "scene_count", "info", "actual_seconds")
    FUNCTION     = "calc"
    CATEGORY     = "LTXAVTools/calculators"

    def calc(self, scene_seconds, fps):
        parts = [p.strip() for p in scene_seconds.replace(",", "|").split("|") if p.strip()]
        if not parts:
            raise ValueError("[LTXSceneLengthCalculator] No scene durations provided.")

        px_list = [max(8, int(round(float(p) * fps / 8)) * 8) for p in parts]
        total_latents = sum(px // 8 for px in px_list)
        frame_count   = (total_latents - 1) * 8 + 1
        actual_seconds = frame_count / fps

        scene_lengths = "|".join(str(px) for px in px_list)
        info = " + ".join(f"{px}px ({px / fps:.2f}s)" for px in px_list)
        info = f"{info} = {frame_count} frames ({actual_seconds:.2f}s)"
        print(f"[LTXSceneLengthCalculator] {info}")

        return (scene_lengths, frame_count, len(px_list), info, actual_seconds)


class LTXKeyframePlanner:
    """
    Plans end-anchored keyframe indices from a scene_lengths schedule.

    Travel semantics: the first keyframe (optional) opens the video at frame 0;
    every subsequent keyframe sits at the END of its scene, so each chunk
    generates TOWARD its destination image and the next scene continues from
    the arrived state via the ordinary overlap carry (no conditioning needed at
    scene starts — start-anchoring instead would put the image in the NEXT
    chunk and invite a snap at every seam). The final scene's end is the
    video's end, emitted as -1 (the sampler's from-the-end convention).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "scene_lengths": ("STRING", {
                    "default": "",
                    "tooltip": "Pipe/comma-separated pixel-frame counts per scene "
                               "(multiples of 8) — from the LTX Scene Length "
                               "Calculator, the LTX Video Cut Marker, or by hand.",
                }),
                "include_start": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keyframe at frame 0 — the opening image (I2V-style).",
                }),
                "include_end": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keyframe at -1 — the closing image (end of the "
                               "final scene).",
                }),
                "fps": ("FLOAT", {
                    "default": 25.0, "min": 1.0, "max": 120.0, "step": 0.01,
                    "tooltip": "For the info receipt's timestamps only.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("indices", "count", "info")
    FUNCTION     = "plan"
    CATEGORY     = "LTXAVTools/calculators"
    DESCRIPTION  = (
        "End-anchored keyframe planner: frame 0 opens, each scene travels to a "
        "keyframe at its end, the final scene ends on -1. Wire `indices` to the "
        "AV Looping Sampler's optional_cond_image_indices; `count` is how many "
        "images optional_cond_images must contain, in order."
    )

    def plan(self, scene_lengths, include_start, include_end, fps):
        parts = [p.strip() for p in scene_lengths.replace(",", "|").split("|") if p.strip()]
        if not parts:
            return ("", 0, "no scenes — empty scene_lengths")

        latents = []
        for p in parts:
            try:
                px = int(round(float(p)))
            except ValueError:
                raise ValueError(f"[LTXKeyframePlanner] non-numeric scene entry '{p}'")
            latents.append(max(1, int(round(px / 8.0))))

        total_latents = sum(latents)
        frame_count   = (total_latents - 1) * 8 + 1

        entries = []   # (index, label_time)
        if include_start:
            entries.append((0, 0.0))
        cum = 0
        for n in latents[:-1]:              # ends of scenes 1..N-1
            cum += n
            end_px = 8 * (cum - 1)          # last pixel frame of the scene
            entries.append((end_px, end_px / fps))
        if include_end:                     # final scene's end = video end
            entries.append((-1, (frame_count - 1) / fps))

        indices = ",".join(str(i) for i, _ in entries)
        info = " · ".join(
            f"img{k} @ {i} ({t:.2f}s{', end' if i == -1 else ''})"
            for k, (i, t) in enumerate(entries)
        )
        info = f"{len(entries)} keyframes for {len(latents)} scenes: {info}"
        print(f"[LTXKeyframePlanner] {info}")

        return (indices, len(entries), info)


NODE_CLASS_MAPPINGS = {
    "LTXDimensionCalculator":       LTXDimensionCalculator,
    "LTXDimensionCalculator3Stage": LTXDimensionCalculator3Stage,
    "LTXFrameCalculator":           LTXFrameCalculator,
    "LTXSceneLengthCalculator":     LTXSceneLengthCalculator,
    "LTXKeyframePlanner":           LTXKeyframePlanner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXDimensionCalculator":       "LTX Dimension Calculator",
    "LTXDimensionCalculator3Stage": "LTX Dimension Calculator 3 Stage",
    "LTXFrameCalculator":           "LTX Frame Calculator",
    "LTXSceneLengthCalculator":     "LTX Scene Length Calculator",
    "LTXKeyframePlanner":           "LTX Keyframe Planner",
}
