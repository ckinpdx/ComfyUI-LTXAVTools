"""
LTX Video Cut Marker — interactive timeline for marking scene boundaries on a
video, emitting the AV Looping Sampler's `scene_lengths` schedule.

The node's web widget (web/js/ltx_video_cut_marker.js) plays the uploaded
video and shows a timeline snapped to the LTX latent grid. Each cut marks the
START of a new scene; scene lengths are the gaps between cuts (plus the final
stretch to the video's end), emitted as pipe-separated pixel-frame counts —
multiples of 8, exactly what the sampler's `scene_lengths` input parses.

Cuts are time-anchored and computed in emit_fps frame space (default 25), so a
24fps source marked while emitting at 25 (the VHS force_rate case) yields a
schedule in the PIPELINE's frame space, not the file's.

Outputs:
- scene_lengths (STRING)  -> sampler `scene_lengths`
- frame_count (INT)       -> total pixel frames (sum - 7), the matching empty-
                             latent length, same pairing convention as the
                             LTX Scene Length Calculator
- video_path (STRING)     -> VHS Load Video (Path); this node does not decode
"""

import os

import folder_paths


def _list_input_videos():
    input_dir = folder_paths.get_input_directory()
    try:
        files = [
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
    except OSError:
        files = []
    try:
        # Core content-type filter when available (matches VHS/core LoadVideo).
        # Audio files are first-class citizens: cutting scenes against a song
        # is the music-video planning case (timeline math is time-anchored, so
        # an audio file works identically — the schedule sizes the video to
        # generate over it).
        files = folder_paths.filter_files_content_types(files, ["video", "audio"])
    except AttributeError:
        exts = (".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v", ".gif",
                ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus")
        files = [f for f in files if f.lower().endswith(exts)]
    return sorted(files)


class LTXVideoCutMarker:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": (_list_input_videos(), {
                    "tooltip": "Video or audio file from the ComfyUI input directory "
                               "(audio = cut scenes against a song; the timeline shows "
                               "its waveform). Use the widget's Upload button or "
                               "drag-and-drop.",
                }),
                "scene_lengths": ("STRING", {
                    "default": "",
                    "tooltip": "Pipe-separated pixel-frame counts per scene "
                               "(multiples of 8), including the final scene to the "
                               "video's end. Maintained by the timeline widget; "
                               "hand-editable.",
                }),
                "emit_fps": ("FLOAT", {
                    "default": 25.0, "min": 1.0, "max": 120.0, "step": 0.01,
                    "tooltip": "Frame space the schedule is emitted in — match the "
                               "rate the video is CONSUMED at downstream (25 for the "
                               "LTX AV pipeline / VHS force_rate 25), not the file's "
                               "native fps. The widget shows the file's detected fps "
                               "for reference only.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "INT")
    RETURN_NAMES = ("scene_lengths", "frame_count", "video_path", "frame_load_cap")
    FUNCTION     = "mark"
    CATEGORY     = "LTXAVTools/utils"
    DESCRIPTION  = (
        "Interactive timeline for marking scene boundaries (and an optional end "
        "marker) on a video, snapped to the LTX latent grid. Emits the "
        "pipe-separated scene_lengths schedule for the AV Looping Sampler, the "
        "matching total frame_count (sum - 7), the video's path (for VHS Load "
        "Video (Path)), and frame_load_cap — equal to frame_count by construction "
        "— for the upstream loader so exactly the scheduled frames are loaded."
    )

    @classmethod
    def IS_CHANGED(s, video, scene_lengths, emit_fps):
        return f"{video}|{scene_lengths}|{emit_fps}"

    def mark(self, video, scene_lengths, emit_fps):
        video_path = folder_paths.get_annotated_filepath(video)

        # Normalize: ints, snapped to multiples of 8, zero-length scenes dropped.
        lengths = []
        for tok in scene_lengths.replace(",", "|").split("|"):
            tok = tok.strip()
            if not tok:
                continue
            try:
                v = int(round(float(tok)))
            except ValueError:
                print(f"[LTXVideoCutMarker] ignoring non-numeric scene entry '{tok}'")
                continue
            v = max(0, int(round(v / 8.0)) * 8)
            if v > 0:
                lengths.append(v)

        out = "|".join(str(v) for v in lengths)
        frame_count = (sum(lengths) - 7) if lengths else 0

        print(f"[LTXVideoCutMarker] scene_lengths: '{out}' | frame_count/load_cap: "
              f"{frame_count} | video: {video_path}")
        return (out, frame_count, video_path, frame_count)


NODE_CLASS_MAPPINGS = {
    "LTXVideoCutMarker": LTXVideoCutMarker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVideoCutMarker": "LTX Video Cut Marker (Scenes)",
}
