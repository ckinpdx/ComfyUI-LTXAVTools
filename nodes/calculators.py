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
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, ratio, orientation, resolution):
        return True

    RETURN_TYPES  = ("INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES  = ("width", "height", "width_half", "height_half", "label")
    FUNCTION      = "calculate"

    def calculate(self, ratio: str, orientation: str, resolution: str):
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
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, ratio, orientation, resolution):
        return True

    RETURN_TYPES  = ("INT", "INT", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES  = ("width", "height", "width_half", "height_half", "width_quarter", "height_quarter", "label")
    FUNCTION      = "calculate"

    def calculate(self, ratio: str, orientation: str, resolution: str):
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


NODE_CLASS_MAPPINGS = {
    "LTXDimensionCalculator":       LTXDimensionCalculator,
    "LTXDimensionCalculator3Stage": LTXDimensionCalculator3Stage,
    "LTXFrameCalculator":           LTXFrameCalculator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXDimensionCalculator":       "LTX Dimension Calculator",
    "LTXDimensionCalculator3Stage": "LTX Dimension Calculator 3 Stage",
    "LTXFrameCalculator":           "LTX Frame Calculator",
}
