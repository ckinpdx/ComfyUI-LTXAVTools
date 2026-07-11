import json
import os
import random
from datetime import datetime


# ---------------------------------------------------------------------------
# Camera / scene vocabulary
# ---------------------------------------------------------------------------

HORIZONTAL_ANGLES = [
    ("front",          "shot from directly in front, subject fully facing the camera"),
    ("front_3q_left",  "three-quarter view, camera positioned to the subject's front-left"),
    ("left_profile",   "pure left-side profile, camera 90 degrees to the subject's left"),
    ("rear_3q_left",   "three-quarter rear view, camera to the subject's rear-left"),
    ("back",           "shot from directly behind, subject's back fully to the camera"),
    ("rear_3q_right",  "three-quarter rear view, camera to the subject's rear-right"),
    ("right_profile",  "pure right-side profile, camera 90 degrees to the subject's right"),
    ("front_3q_right", "three-quarter view, camera positioned to the subject's front-right"),
]

VERTICAL_ANGLES = [
    ("eye_level",  "camera at eye level"),
    ("low_angle",  "low camera angle, camera below eye level looking upward at the subject"),
    ("high_angle", "high camera angle, camera above eye level looking downward at the subject"),
]

FRAMINGS = [
    ("full_body",      "full-body shot, head to toe"),
    ("three_quarter",  "three-quarter shot, head to mid-thigh"),
    ("waist_up",       "waist-up shot"),
    ("bust",           "bust shot, head and chest"),
    ("head_shoulders", "head-and-shoulders portrait"),
    ("face_closeup",   "face close-up, face filling the frame"),
]

# Rear-facing angles cannot show a face closeup
_INVALID = {
    ("back",          "face_closeup"),
    ("rear_3q_left",  "face_closeup"),
    ("rear_3q_right", "face_closeup"),
}

NEUTRAL_POSE = (
    "standing in a relaxed neutral pose, arms hanging naturally at the sides, "
    "feet shoulder-width apart, looking straight ahead, neutral facial expression, "
    "no action, no gesture"
)
BACKGROUND = "plain white seamless paper backdrop, white cyclorama, pure white studio background"
LIGHTING   = (
    "soft even studio lighting, large softbox lights, no harsh shadows, "
    "uniform flat illumination, shadowless"
)


def _build_combinations():
    combos = []
    for h_key, h_desc in HORIZONTAL_ANGLES:
        for v_key, v_desc in VERTICAL_ANGLES:
            for f_key, f_desc in FRAMINGS:
                if (h_key, f_key) in _INVALID:
                    continue
                combos.append({
                    "key":    f"{h_key}__{v_key}__{f_key}",
                    "h_desc": h_desc,
                    "v_desc": v_desc,
                    "f_desc": f_desc,
                })
    return combos


ALL_COMBINATIONS = _build_combinations()
TOTAL            = len(ALL_COMBINATIONS)

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "character_data")


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class CharacterDatasetPromptGenerator:
    """
    Generates non-repeating camera-angle prompts for character LoRA dataset creation.
    Persists used-prompt history per character across workflow runs.
    Cycles back through all angles once every combination has been used.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_name": ("STRING", {
                    "default": "my_character",
                    "multiline": False,
                    "tooltip": "Used as the filename for the tracking JSON. One file per character.",
                }),
                "character_description": ("STRING", {
                    "default": "a person",
                    "multiline": True,
                    "tooltip": "Appearance description inserted into every prompt.",
                }),
                "mode": (["sequential", "random"], {
                    "tooltip": "sequential picks combinations in order; random uses seed + run count.",
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Only used in random mode.",
                }),
                "reset": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Clear the history for this character and start over.",
                }),
            },
        }

    RETURN_TYPES  = ("STRING",  "INT",        "INT",    "STRING")
    RETURN_NAMES  = ("prompt",  "remaining",  "total",  "view_key")
    FUNCTION      = "generate"
    CATEGORY      = "LTXAVTools/dataset"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # never cache — always execute

    # ------------------------------------------------------------------

    def _data_path(self, character_name: str) -> str:
        os.makedirs(_DATA_DIR, exist_ok=True)
        safe = "".join(c for c in character_name if c.isalnum() or c in "-_ ")
        safe = safe.strip().replace(" ", "_").lower() or "character"
        return os.path.join(_DATA_DIR, f"{safe}.json")

    def _load(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"used": [], "total_generated": 0}

    def _save(self, path: str, data: dict):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def generate(self, character_name, character_description, mode, seed=0, reset=False):
        path = self._data_path(character_name)
        data = self._load(path)

        if reset:
            data["used"] = []
            data["total_generated"] = 0

        used_keys = set(data["used"])
        available = [c for c in ALL_COMBINATIONS if c["key"] not in used_keys]

        # Wrap around when every combination has been used
        if not available:
            data["used"] = []
            data["total_generated"] = 0
            available = list(ALL_COMBINATIONS)
            print(f"[CharacterDatasetPrompt] All {TOTAL} views used for '{character_name}' — cycling.")

        if mode == "random":
            rng   = random.Random(seed + data["total_generated"])
            combo = rng.choice(available)
        else:
            combo = available[0]

        prompt = ", ".join([
            combo["f_desc"],
            character_description.strip(),
            NEUTRAL_POSE,
            combo["h_desc"],
            combo["v_desc"],
            BACKGROUND,
            LIGHTING,
        ])

        data["used"].append(combo["key"])
        data["total_generated"] += 1
        data["last_view"] = combo["key"]
        data["last_run"]  = datetime.now().isoformat()
        self._save(path, data)

        remaining = len(available) - 1
        print(f"[CharacterDatasetPrompt] '{character_name}' — {combo['key']} | {remaining} remaining of {TOTAL}")
        return (prompt, remaining, TOTAL, combo["key"])


NODE_CLASS_MAPPINGS = {
    "CharacterDatasetPromptGenerator": CharacterDatasetPromptGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CharacterDatasetPromptGenerator": "Character Dataset Prompt Generator",
}