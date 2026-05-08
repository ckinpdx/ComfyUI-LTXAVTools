import json
import os
import re
import shutil
import subprocess
import sys
import time
import gc
import math
from datetime import datetime

import comfy
import torch
import cv2
import numpy as np
import folder_paths

try:
    import torchaudio
except Exception:
    torchaudio = None


class _LTXLoraTrainBase:
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
    VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
    AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a", ".opus"}

    RETURN_TYPES = ("MODEL", "STRING", "STRING", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = (
        "model",
        "latest_state_path",
        "log_path",
        "video_filename_prefix",
        "output_name",
        "completed_steps",
        "total_target_steps",
    )
    FUNCTION = "run"
    CATEGORY = "LTXAVTools/Training"

    @staticmethod
    def _norm(path):
        return os.path.normpath(str(path or "").strip())

    @staticmethod
    def _safe_name(value, default_value):
        raw = str(value or "").strip() or default_value
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
        return cleaned.strip("._-") or default_value

    @staticmethod
    def _ensure_dir(path):
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def _quote(path):
        return path.replace("\\", "/")

    @staticmethod
    def _parse_step(name):
        match = re.search(r"step(\d+)", name or "")
        return int(match.group(1)) if match else 0

    @staticmethod
    def _format_duration(seconds):
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes}m {seconds}s"
        if minutes:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    def _print_stage_banner(self, log_handle, stage_number, total_stages, title, detail_lines=None):
        lines = ["", "=" * 78, f"[LTXAVTools] STAGE {stage_number}/{total_stages}: {title}"]
        if detail_lines:
            lines.extend(f"[LTXAVTools] {line}" for line in detail_lines)
        lines.append("=" * 78)
        lines.append("")
        banner = "\n".join(lines)
        print(banner)
        log_handle.write(banner + "\n")
        log_handle.flush()

    def _run_stage_command(self, stage_number, total_stages, title, command, cwd, log_handle, detail_lines=None):
        self._print_stage_banner(log_handle, stage_number, total_stages, title, detail_lines)
        started_at = time.time()
        self._run_command(command, cwd, log_handle)
        elapsed = self._format_duration(time.time() - started_at)
        completion_line = f"[LTXAVTools] Completed stage {stage_number}/{total_stages}: {title} in {elapsed}"
        print(completion_line)
        log_handle.write(completion_line + "\n")
        log_handle.flush()

    @staticmethod
    def _extract_command_exit_code(error):
        match = re.search(r"exit code (\d+)", str(error))
        return int(match.group(1)) if match else None

    def _build_text_encoder_cache_command(
        self, python_exe, dataset_config, ltx2_checkpoint, gemma_root, ltx_mode,
        gemma_load_in_8bit, mixed_precision="bf16", gemma_load_in_4bit=False,
    ):
        command = [
            python_exe, "ltx2_cache_text_encoder_outputs.py",
            "--dataset_config", dataset_config,
            "--ltx2_checkpoint", ltx2_checkpoint,
            "--gemma_root", gemma_root,
        ]
        if gemma_load_in_4bit:
            command.append("--gemma_load_in_4bit")
        elif gemma_load_in_8bit:
            command.append("--gemma_load_in_8bit")
        command.extend([
            "--device", "cuda",
            "--mixed_precision", str(mixed_precision or "bf16"),
            "--ltx2_mode", ltx_mode,
            "--batch_size", "1",
        ])
        return command

    def _run_text_encoder_cache_stage(
        self, stage_number, total_stages, title, python_exe, dataset_config,
        ltx2_checkpoint, gemma_root, ltx_mode, gemma_load_in_8bit, cwd, log_handle,
        detail_lines=None, gemma_load_in_4bit=False,
    ):
        command = self._build_text_encoder_cache_command(
            python_exe, dataset_config, ltx2_checkpoint, gemma_root, ltx_mode,
            gemma_load_in_8bit, gemma_load_in_4bit=gemma_load_in_4bit,
        )
        self._run_stage_command(stage_number, total_stages, title, command, cwd, log_handle, detail_lines)

    def _run_text_encoder_cache_stage_with_recovery(
        self, stage_number, total_stages, title, python_exe, dataset_config,
        ltx2_checkpoint, gemma_root, ltx_mode, gemma_load_in_8bit, recovery_mode,
        cwd, log_handle, detail_lines=None, gemma_load_in_4bit=False,
    ):
        try:
            self._run_text_encoder_cache_stage(
                stage_number, total_stages, title, python_exe, dataset_config,
                ltx2_checkpoint, gemma_root, ltx_mode, gemma_load_in_8bit, cwd,
                log_handle, detail_lines, gemma_load_in_4bit=gemma_load_in_4bit,
            )
            return
        except RuntimeError as exc:
            if not recovery_mode:
                raise
            last_exc = exc

        gemma_flag_label = "--gemma_load_in_4bit" if gemma_load_in_4bit else "--gemma_load_in_8bit"
        retry_plans = [
            ("fp16", f"experimental recovery: retrying with mixed_precision=fp16 and {gemma_flag_label}"),
            ("bf16", f"experimental recovery: retrying with mixed_precision=bf16 and {gemma_flag_label}"),
        ]
        for mixed_precision, label in retry_plans:
            retry_message = f"[LTXAVTools] {title} failed. {label}."
            print(retry_message)
            log_handle.write(retry_message + "\n")
            log_handle.flush()
            retry_command = self._build_text_encoder_cache_command(
                python_exe, dataset_config, ltx2_checkpoint, gemma_root, ltx_mode,
                gemma_load_in_8bit, mixed_precision, gemma_load_in_4bit=gemma_load_in_4bit,
            )
            try:
                self._run_stage_command(stage_number, total_stages, title, retry_command, cwd, log_handle, detail_lines)
                return
            except RuntimeError as exc:
                last_exc = exc

        raise RuntimeError(
            f"{title} failed after experimental recovery attempts. "
            "Check the log for the exact Gemma cache command outputs."
        ) from last_exc

    def _count_dataset_files(self, images_dir):
        image_count = 0
        caption_count = 0
        if not os.path.isdir(images_dir):
            return image_count, caption_count
        for entry in os.scandir(images_dir):
            if not entry.is_file():
                continue
            ext = os.path.splitext(entry.name)[1].lower()
            if ext in self.IMAGE_EXTENSIONS:
                image_count += 1
            elif ext == ".txt":
                caption_count += 1
        return image_count, caption_count

    def _count_cache_files(self, cache_dir):
        if not os.path.isdir(cache_dir):
            return 0
        file_count = 0
        for _, _, files in os.walk(cache_dir):
            file_count += len(files)
        return file_count

    def _has_latent_cache_files(self, cache_dir):
        architecture = "ltx2"
        if not os.path.isdir(cache_dir):
            return False
        for entry in os.scandir(cache_dir):
            if not entry.is_file():
                continue
            if (entry.name.endswith(f"_{architecture}.safetensors")
                    and not entry.name.endswith(f"_{architecture}_te.safetensors")
                    and not entry.name.endswith(f"_{architecture}_audio.safetensors")):
                return True
        return False

    def _has_complete_text_encoder_cache_files(self, cache_dir):
        architecture = "ltx2"
        if not os.path.isdir(cache_dir):
            return False

        latent_cache_files = []
        for entry in os.scandir(cache_dir):
            if not entry.is_file():
                continue
            if (entry.name.endswith(f"_{architecture}.safetensors")
                    and not entry.name.endswith(f"_{architecture}_te.safetensors")
                    and not entry.name.endswith(f"_{architecture}_audio.safetensors")):
                latent_cache_files.append(entry.path)

        if not latent_cache_files:
            return False

        seen_text_cache_files = set()
        for latent_cache_file in latent_cache_files:
            expected_text_cache = self._expected_text_cache_path_from_latent_cache_file(cache_dir, latent_cache_file)
            if not expected_text_cache:
                continue
            normalized_expected = os.path.normpath(expected_text_cache)
            if normalized_expected in seen_text_cache_files:
                continue
            seen_text_cache_files.add(normalized_expected)
            if not os.path.exists(expected_text_cache):
                return False
        return True

    def _expected_text_cache_path_from_latent_cache_file(self, cache_dir, latent_cache_file):
        architecture = "ltx2"
        basename = os.path.basename(latent_cache_file)
        if not basename.endswith(f"_{architecture}.safetensors"):
            return ""
        tokens = basename.split("_")
        if len(tokens) < 4:
            return ""
        if len(tokens) >= 5 and re.fullmatch(r"\d+-\d+(?:-\d+)?", tokens[-3]):
            item_key = "_".join(tokens[:-3])
        else:
            item_key = "_".join(tokens[:-2])
        if not item_key:
            return ""
        return os.path.join(cache_dir, f"{item_key}_{architecture}_te.safetensors")

    def _get_dataset_label(self, dataset_images_dir):
        dataset_images_dir = self._norm(dataset_images_dir)
        base_name = os.path.basename(dataset_images_dir)
        if base_name.lower() == "images":
            parent = os.path.dirname(dataset_images_dir)
            base_name = os.path.basename(parent) or base_name
        return self._safe_name(base_name, "dataset")

    def _get_or_create_video_output_subfolder(self, config_dir, dataset_images_dir):
        subfolder_file = os.path.join(config_dir, "video_output_subfolder.txt")
        if os.path.isfile(subfolder_file):
            with open(subfolder_file, "r", encoding="utf-8") as handle:
                existing_subfolder = handle.read().strip()
            if existing_subfolder:
                return existing_subfolder

        dataset_label = self._get_dataset_label(dataset_images_dir)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        relative_subfolder = f"{dataset_label}_{stamp}"

        with open(subfolder_file, "w", encoding="utf-8") as handle:
            handle.write(relative_subfolder)

        output_root = folder_paths.get_output_directory()
        os.makedirs(os.path.join(output_root, relative_subfolder), exist_ok=True)
        return relative_subfolder

    def _build_video_filename_prefix(self, video_output_subfolder, output_name, step_number):
        return f"{video_output_subfolder}/{self._safe_name(output_name, 'LTXChunkRun')}_step_{int(step_number)}"

    def _get_or_create_video_filename_prefix(self, config_dir, dataset_images_dir, output_name, step_number):
        prefix_file = os.path.join(config_dir, "video_filename_prefix.txt")
        current_prefix = self._build_video_filename_prefix(
            self._get_or_create_video_output_subfolder(config_dir, dataset_images_dir),
            output_name,
            step_number,
        )
        with open(prefix_file, "w", encoding="utf-8") as handle:
            handle.write(current_prefix)
        return current_prefix

    def _clear_memory_before_gemma(self, log_handle):
        messages = ["[LTXAVTools] Clearing ComfyUI and CUDA memory before text encoder cache stage."]
        try:
            comfy.model_management.unload_all_models()
            messages.append("[LTXAVTools] unload_all_models() completed.")
        except Exception as exc:
            messages.append(f"[LTXAVTools] unload_all_models() skipped: {exc}")
        try:
            comfy.model_management.cleanup_models()
            messages.append("[LTXAVTools] cleanup_models() completed.")
        except Exception as exc:
            messages.append(f"[LTXAVTools] cleanup_models() skipped: {exc}")
        try:
            comfy.model_management.soft_empty_cache(force=True)
            messages.append("[LTXAVTools] soft_empty_cache(force=True) completed.")
        except Exception as exc:
            messages.append(f"[LTXAVTools] soft_empty_cache(force=True) skipped: {exc}")

        gc.collect()
        messages.append("[LTXAVTools] Python garbage collection completed.")

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                messages.append("[LTXAVTools] torch.cuda.empty_cache() completed.")
            except Exception as exc:
                messages.append(f"[LTXAVTools] torch.cuda.empty_cache() skipped: {exc}")
            try:
                torch.cuda.ipc_collect()
                messages.append("[LTXAVTools] torch.cuda.ipc_collect() completed.")
            except Exception as exc:
                messages.append(f"[LTXAVTools] torch.cuda.ipc_collect() skipped: {exc}")

        for message in messages:
            print(message)
            log_handle.write(message + "\n")
        log_handle.flush()

    @staticmethod
    def _resolve_learning_rate(learning_rate_preset, learning_rate):
        preset = str(learning_rate_preset or "Custom").strip()
        if preset and preset != "Custom":
            return float(preset)
        return float(learning_rate)

    def _latest_state_dir(self, output_dir, output_name):
        if not os.path.isdir(output_dir):
            return "", 0
        prefix = f"{output_name}-step"
        candidates = []
        for entry in os.scandir(output_dir):
            if not entry.is_dir():
                continue
            if not entry.name.startswith(prefix) or not entry.name.endswith("-state"):
                continue
            step = self._parse_step(entry.name)
            candidates.append((step, entry.path))
        if not candidates:
            return "", 0
        step, path = max(candidates, key=lambda item: item[0])
        return os.path.normpath(path), step

    def _latest_file(self, output_dir, output_name, suffix):
        if not os.path.isdir(output_dir):
            return "", 0
        prefix = f"{output_name}-step"
        candidates = []
        for entry in os.scandir(output_dir):
            if not entry.is_file():
                continue
            if not entry.name.startswith(prefix) or not entry.name.endswith(suffix):
                continue
            if suffix == ".safetensors" and entry.name.endswith(".comfy.safetensors"):
                continue
            step = self._parse_step(entry.name)
            candidates.append((step, entry.path))
        if not candidates:
            return "", 0
        step, path = max(candidates, key=lambda item: item[0])
        return os.path.normpath(path), step

    def _write_dataset_config(self, path, dataset_images_dir, cache_dir, width, height, num_repeats):
        content = (
            "[general]\n"
            f"resolution = [{int(width)}, {int(height)}]\n"
            'caption_extension = ".txt"\n'
            "batch_size = 1\n"
            "enable_bucket = true\n"
            "bucket_no_upscale = false\n\n"
            "[[datasets]]\n"
            f'image_directory = "{self._quote(dataset_images_dir)}"\n'
            f'cache_directory = "{self._quote(cache_dir)}"\n'
            f"num_repeats = {int(num_repeats)}\n"
        )
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)

    def _write_training_config(
        self, path, dataset_config, checkpoint, gemma_root, output_dir, log_dir,
        output_name, network_dim, network_alpha, blocks_to_swap, learning_rate,
        max_train_steps, steps_per_run, total_target_steps, lora_target_preset="full",
    ):
        content = (
            "# Auto-generated by LTXAVTools LTX chunk trainer\n"
            f"# total_target_steps_from_workflow = {int(total_target_steps)}\n"
            f"# chunk_target_steps_this_run = {int(max_train_steps)}\n"
            f"# save_interval_per_run = {int(steps_per_run)}\n"
            f'ltx2_checkpoint = "{self._quote(checkpoint)}"\n'
            f'gemma_root = "{self._quote(gemma_root)}"\n'
            f'dataset_config = "{self._quote(dataset_config)}"\n\n'
            'ltx_mode = "video"\n'
            'ltx_version = "2.3"\n'
            'ltx_version_check_mode = "error"\n'
            f'lora_target_preset = "{lora_target_preset}"\n\n'
            "cache_text_encoder_outputs = true\n"
            "cache_text_encoder_outputs_to_disk = false\n\n"
            "fp8_base = true\n"
            "fp8_scaled = true\n"
            "sdpa = true\n"
            "gradient_checkpointing = true\n"
            "gradient_accumulation_steps = 1\n"
            f"blocks_to_swap = {int(blocks_to_swap)}\n\n"
            'optimizer_type = "AdamW8Bit"\n'
            f"learning_rate = {learning_rate}\n"
            'lr_scheduler = "constant_with_warmup"\n'
            "lr_warmup_steps = 100\n\n"
            'network_module = "networks.lora_ltx2"\n'
            f"network_dim = {int(network_dim)}\n"
            f"network_alpha = {int(network_alpha)}\n"
            'timestep_sampling = "shifted_logit_normal"\n'
            "ltx2_first_frame_conditioning_p = 0.5\n\n"
            f'output_dir = "{self._quote(output_dir)}"\n'
            f'output_name = "{output_name}"\n'
            'log_with = "tensorboard"\n'
            f'logging_dir = "{self._quote(log_dir)}"\n'
            "log_config = true\n"
            f"max_train_steps = {int(max_train_steps)}\n"
            f"save_every_n_steps = {int(steps_per_run)}\n"
            'save_model_as = "safetensors"\n'
            'mixed_precision = "bf16"\n'
            "save_state = true\n"
            "save_state_on_train_end = true\n"
        )
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)

    def _resolve_musubi_executables(self, musubi_root):
        checked = []
        for env_name in (".venv", "venv", "env"):
            scripts_dir = os.path.join(musubi_root, env_name, "Scripts")
            python_candidate = os.path.join(scripts_dir, "python.exe")
            accelerate_candidate = os.path.join(scripts_dir, "accelerate.exe")
            checked.append((python_candidate, accelerate_candidate))
            if os.path.isfile(python_candidate) and os.path.isfile(accelerate_candidate):
                return os.path.normpath(python_candidate), os.path.normpath(accelerate_candidate), env_name

        current_python = os.path.normpath(sys.executable)
        current_scripts_dir = os.path.dirname(current_python)
        for accelerate_name in ("accelerate.exe", "accelerate"):
            accelerate_candidate = os.path.join(current_scripts_dir, accelerate_name)
            checked.append((current_python, accelerate_candidate))
            if os.path.isfile(current_python) and os.path.isfile(accelerate_candidate):
                return current_python, os.path.normpath(accelerate_candidate), "current_python_env"

        path_python = shutil.which("python")
        path_accelerate = shutil.which("accelerate")
        checked.append((path_python or "(not found)", path_accelerate or "(not found)"))
        if path_python and path_accelerate:
            return os.path.normpath(path_python), os.path.normpath(path_accelerate), "PATH"

        checked_lines = "\n".join(
            f"  python={p}\n  accelerate={a}" for p, a in checked
        )
        raise ValueError(
            "Could not resolve musubi Python/accelerate executables.\n"
            f"Checked:\n{checked_lines}\n"
            "Supported layouts: .venv, venv, env, or current/PATH environment."
        )

    def _resolve_musubi_script_root(self, musubi_root, required_scripts):
        musubi_root = self._norm(musubi_root)
        required_scripts = [str(s or "").strip() for s in required_scripts if str(s or "").strip()]
        if not required_scripts:
            return musubi_root

        def _has_required(root):
            return os.path.isdir(root) and all(os.path.isfile(os.path.join(root, s)) for s in required_scripts)

        if _has_required(musubi_root):
            return musubi_root

        parent_root = os.path.dirname(musubi_root)
        if os.path.isdir(parent_root):
            for entry in os.scandir(parent_root):
                if not entry.is_dir():
                    continue
                if _has_required(entry.path):
                    print(f"[LTXAVTools] Resolved Musubi script root to {entry.path}")
                    return os.path.normpath(entry.path)

        missing = ", ".join(required_scripts)
        raise ValueError(f"musubi_root does not contain required scripts: {missing}. Checked: {musubi_root}")

    def _run_command(self, command, cwd, log_handle):
        command_line = f"$ {' '.join(command)}"
        log_handle.write(command_line + "\n")
        log_handle.flush()
        print(command_line)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        process = subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, universal_newlines=True, encoding="utf-8", errors="replace", env=env,
        )
        output_lines = []
        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()
            print(line, end="")
            output_lines.append(line.rstrip("\n"))
        process.wait()
        if process.returncode != 0:
            tail = "\n".join(output_lines[-40:]).strip()
            message = f"Command failed with exit code {process.returncode}: {' '.join(command)}"
            if tail:
                message += f"\n--- output tail ---\n{tail}"
            raise RuntimeError(message)

    def _should_build_cache(self, cache_strategy, cache_dir):
        if cache_strategy == "force":
            return True
        if cache_strategy == "skip":
            return False
        if not os.path.isdir(cache_dir):
            return True
        return not (self._has_latent_cache_files(cache_dir) and self._has_complete_text_encoder_cache_files(cache_dir))

    def _export_latest_to_comfy(self, latest_comfy_lora, output_name):
        if not latest_comfy_lora or not os.path.isfile(latest_comfy_lora):
            return ""
        lora_dirs = folder_paths.get_folder_paths("loras")
        if not lora_dirs:
            raise RuntimeError("ComfyUI loras folder could not be resolved.")
        target_dir = lora_dirs[0]
        self._ensure_dir(target_dir)
        target_path = os.path.join(target_dir, f"{output_name}_latest.comfy.safetensors")
        shutil.copy2(latest_comfy_lora, target_path)
        return os.path.normpath(target_path)

    def _delete_standard_lora_files(self, output_dir, output_name):
        if not os.path.isdir(output_dir):
            return 0
        prefix = f"{output_name}-step"
        deleted_count = 0
        for entry in os.scandir(output_dir):
            if not entry.is_file():
                continue
            if not entry.name.startswith(prefix):
                continue
            if not entry.name.endswith(".safetensors"):
                continue
            if entry.name.endswith(".comfy.safetensors"):
                continue
            os.remove(entry.path)
            deleted_count += 1
        return deleted_count

    def _apply_lora_to_model(self, model, lora_path, strength_model):
        if not lora_path or not os.path.isfile(lora_path):
            return model
        strength = float(strength_model)
        if strength == 0:
            return model
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        model_lora, _ = comfy.sd.load_lora_for_models(model, None, lora, strength, 0)
        return model_lora

    def _log_message(self, message, log_path=None):
        print(message)
        if log_path:
            try:
                with open(log_path, "a", encoding="utf-8") as handle:
                    handle.write(message + "\n")
            except Exception:
                pass

    def _compose_caption_text(self, base_caption, add_trigger_word, trigger_text):
        caption = str(base_caption or "").strip()
        trigger = str(trigger_text or "").strip()
        if not add_trigger_word or not trigger:
            return caption
        if caption:
            if caption == trigger or caption.startswith(f"{trigger},") or caption.startswith(f"{trigger} "):
                return caption
            return f"{trigger}, {caption}"
        return trigger

    def _ensure_captions(self, images_dir, create_captions, caption_text, add_trigger_word, trigger_text):
        image_entries = [
            entry for entry in os.scandir(images_dir)
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in self.IMAGE_EXTENSIONS
        ]
        base_caption = str(caption_text or "").strip()
        created_count = 0
        updated_count = 0
        for entry in image_entries:
            stem = os.path.splitext(entry.name)[0]
            caption_path = os.path.join(images_dir, f"{stem}.txt")
            existing_caption = ""
            if os.path.isfile(caption_path):
                with open(caption_path, "r", encoding="utf-8") as handle:
                    existing_caption = handle.read().strip()
            elif not create_captions:
                continue
            caption_body = existing_caption if existing_caption else base_caption
            final_caption = self._compose_caption_text(caption_body, add_trigger_word, trigger_text)
            if not existing_caption and not final_caption:
                continue
            if existing_caption == final_caption:
                continue
            with open(caption_path, "w", encoding="utf-8") as handle:
                handle.write(final_caption)
            if existing_caption:
                updated_count += 1
            else:
                created_count += 1
        print(
            f"[LTXAVTools] Caption prep complete. created={created_count} updated={updated_count} "
            f"trigger={'on' if add_trigger_word else 'off'}"
        )

    def _prepare_dataset_directory(self, dataset_root, create_captions, caption_text, add_trigger_word, trigger_text):
        dataset_root = self._norm(dataset_root)
        if not os.path.isdir(dataset_root):
            raise ValueError(f"dataset_images_dir does not exist: {dataset_root}")

        if os.path.basename(dataset_root).lower() == "images":
            self._ensure_captions(dataset_root, create_captions, caption_text, add_trigger_word, trigger_text)
            return dataset_root

        images_dir = os.path.join(dataset_root, "images")
        if os.path.isdir(images_dir):
            self._ensure_captions(images_dir, create_captions, caption_text, add_trigger_word, trigger_text)
            return os.path.normpath(images_dir)

        os.makedirs(images_dir, exist_ok=True)
        root_files = [entry for entry in os.scandir(dataset_root) if entry.is_file()]
        image_stems = {
            os.path.splitext(entry.name)[0]
            for entry in root_files
            if os.path.splitext(entry.name)[1].lower() in self.IMAGE_EXTENSIONS
        }
        moved_count = 0
        for entry in root_files:
            ext = os.path.splitext(entry.name)[1].lower()
            stem = os.path.splitext(entry.name)[0]
            should_move = ext in self.IMAGE_EXTENSIONS or (ext == ".txt" and stem in image_stems)
            if not should_move:
                continue
            target_path = os.path.join(images_dir, entry.name)
            if os.path.exists(target_path):
                continue
            shutil.move(entry.path, target_path)
            moved_count += 1
        self._ensure_captions(images_dir, create_captions, caption_text, add_trigger_word, trigger_text)
        print(f"[LTXAVTools] Dataset prep complete. Using images folder: {images_dir} (moved {moved_count} file(s))")
        return os.path.normpath(images_dir)

    def run(
        self,
        model, dataset_images_dir, workspace_dir, run_name, output_name,
        resolution_width, resolution_height, steps_per_run, total_target_steps,
        network_dim, network_alpha, blocks_to_swap, clear_memory_before_gemma,
        gemma_recovery_mode, learning_rate_preset, learning_rate, num_repeats,
        cache_strategy, copy_latest_to_comfy_loras, keep_only_comfy_lora,
        strength_model, create_captions, caption_text, add_trigger_word,
        trigger_text, musubi_root, ltx2_checkpoint, gemma_root,
        gemma_load_in_4bit=False, lora_target_preset="full", _autochunk_mode=False,
    ):
        dataset_images_dir = self._norm(dataset_images_dir)
        workspace_dir = self._norm(workspace_dir)
        musubi_root = self._norm(musubi_root)
        ltx2_checkpoint = self._norm(ltx2_checkpoint)
        gemma_root = self._norm(gemma_root)
        run_name = self._safe_name(run_name, "LTXChunkRun")
        output_name = self._safe_name(output_name, run_name)
        effective_learning_rate = self._resolve_learning_rate(learning_rate_preset, learning_rate)

        dataset_images_dir = self._prepare_dataset_directory(
            dataset_images_dir, create_captions, caption_text, add_trigger_word, trigger_text,
        )
        workspace_dir = self._ensure_dir(workspace_dir)
        if not os.path.isfile(ltx2_checkpoint):
            raise ValueError(f"ltx2_checkpoint does not exist: {ltx2_checkpoint}")
        if not os.path.isdir(gemma_root):
            raise ValueError(f"gemma_root does not exist: {gemma_root}")

        musubi_root = self._resolve_musubi_script_root(musubi_root, [
            "ltx2_cache_latents.py",
            "ltx2_cache_text_encoder_outputs.py",
            "ltx2_train_network.py",
        ])

        gemma_load_in_4bit = bool(gemma_load_in_4bit)
        gemma_load_in_8bit = not gemma_load_in_4bit
        gemma_load_mode = "4bit" if gemma_load_in_4bit else "8bit"

        python_exe, accelerate_exe, env_source = self._resolve_musubi_executables(musubi_root)

        cache_dir = self._ensure_dir(os.path.join(workspace_dir, "cache"))
        output_dir = self._ensure_dir(os.path.join(workspace_dir, "output"))
        logs_dir = self._ensure_dir(os.path.join(workspace_dir, "logs"))
        config_dir = self._ensure_dir(os.path.join(workspace_dir, "config"))
        dataset_config = os.path.join(config_dir, "dataset-01.toml")
        training_config = os.path.join(config_dir, "training_args.toml")

        latest_state_path, completed_steps = self._latest_state_dir(output_dir, output_name)
        if completed_steps >= int(total_target_steps):
            raise RuntimeError(
                f"Training complete: reached {completed_steps}/{int(total_target_steps)} steps. Stopping workflow."
            )

        next_target_steps = min(completed_steps + int(steps_per_run), int(total_target_steps))
        video_filename_prefix = self._get_or_create_video_filename_prefix(
            config_dir, dataset_images_dir, output_name, next_target_steps,
        )

        self._write_dataset_config(
            dataset_config, dataset_images_dir, cache_dir,
            resolution_width, resolution_height, num_repeats,
        )
        self._write_training_config(
            training_config, dataset_config, ltx2_checkpoint, gemma_root,
            output_dir, logs_dir, output_name, network_dim, network_alpha,
            blocks_to_swap, effective_learning_rate, next_target_steps,
            steps_per_run, total_target_steps, lora_target_preset=lora_target_preset,
        )

        log_path = os.path.join(logs_dir, f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        with open(log_path, "w", encoding="utf-8") as log_handle:
            log_handle.write(f"dataset_images_dir={dataset_images_dir}\n")
            log_handle.write(f"workspace_dir={workspace_dir}\n")
            log_handle.write(f"completed_steps={completed_steps}\n")
            log_handle.write(f"next_target_steps={next_target_steps}\n\n")
            log_handle.flush()

            image_count, caption_count = self._count_dataset_files(dataset_images_dir)
            cache_file_count_before = self._count_cache_files(cache_dir)
            should_build_cache = self._should_build_cache(cache_strategy, cache_dir)
            total_stages = 3 if should_build_cache else 1

            print(f"[LTXAVTools] dataset_images_dir={dataset_images_dir}")
            print(f"[LTXAVTools] workspace_dir={workspace_dir}")
            print(f"[LTXAVTools] video_filename_prefix={video_filename_prefix}")
            print(f"[LTXAVTools] completed_steps={completed_steps}")
            print(f"[LTXAVTools] next_target_steps={next_target_steps}")
            print(f"[LTXAVTools] steps_per_run_and_save={steps_per_run}")
            print(f"[LTXAVTools] total_target_steps={total_target_steps}")
            print(f"[LTXAVTools] blocks_to_swap={int(blocks_to_swap)}")
            print(f"[LTXAVTools] musubi_env_source={env_source}")
            print(f"[LTXAVTools] musubi_python={python_exe}")
            print(f"[LTXAVTools] musubi_accelerate={accelerate_exe}")
            print(f"[LTXAVTools] clear_memory_before_gemma={clear_memory_before_gemma}")
            print(f"[LTXAVTools] gemma_load_mode={gemma_load_mode}")
            print(f"[LTXAVTools] gemma_recovery_mode={gemma_recovery_mode}")
            print(f"[LTXAVTools] keep_only_comfy_lora={keep_only_comfy_lora}")
            print(f"[LTXAVTools] learning_rate={effective_learning_rate} (preset={learning_rate_preset})")
            print(f"[LTXAVTools] dataset summary: images={image_count} captions={caption_count}")
            print(
                f"[LTXAVTools] cache summary: strategy={cache_strategy} "
                f"build_cache={'yes' if should_build_cache else 'no'} "
                f"existing_cache_files={cache_file_count_before}"
            )
            if latest_state_path:
                print(f"[LTXAVTools] resume state detected: {latest_state_path}")
            else:
                print("[LTXAVTools] resume state detected: none")

            if should_build_cache:
                self._run_stage_command(
                    1, total_stages, "Cache latents",
                    [python_exe, "ltx2_cache_latents.py",
                     "--dataset_config", dataset_config,
                     "--ltx2_checkpoint", ltx2_checkpoint,
                     "--device", "cuda", "--vae_dtype", "bf16", "--ltx2_mode", "video"],
                    musubi_root, log_handle,
                    [f"Dataset images dir: {dataset_images_dir}",
                     f"Images found: {image_count}", f"Captions found: {caption_count}",
                     f"Cache dir: {cache_dir}"],
                )
                if clear_memory_before_gemma:
                    self._clear_memory_before_gemma(log_handle)
                if gemma_recovery_mode:
                    self._run_text_encoder_cache_stage_with_recovery(
                        2, total_stages, "Cache text encoder outputs",
                        python_exe, dataset_config, ltx2_checkpoint, gemma_root,
                        "video", gemma_load_in_8bit, gemma_recovery_mode, musubi_root, log_handle,
                        [f"Gemma root: {gemma_root}", "This is usually the slowest setup stage.",
                         "You should see per-item progress from the text encoder cache script."],
                        gemma_load_in_4bit=gemma_load_in_4bit,
                    )
                else:
                    self._run_text_encoder_cache_stage(
                        2, total_stages, "Cache text encoder outputs",
                        python_exe, dataset_config, ltx2_checkpoint, gemma_root,
                        "video", gemma_load_in_8bit, musubi_root, log_handle,
                        [f"Gemma root: {gemma_root}", "This is usually the slowest setup stage.",
                         "You should see per-item progress from the text encoder cache script."],
                        gemma_load_in_4bit=gemma_load_in_4bit,
                    )
                print(f"[LTXAVTools] cache summary after build: files={self._count_cache_files(cache_dir)}")
            else:
                self._print_stage_banner(
                    log_handle, 1, total_stages, "Skip cache build",
                    [f"Cache strategy: {cache_strategy}",
                     f"Existing cache files: {cache_file_count_before}",
                     "Proceeding directly to training."],
                )

            train_command = [
                accelerate_exe, "launch",
                "--num_cpu_threads_per_process", "1",
                "--mixed_precision", "bf16",
                "ltx2_train_network.py",
                "--config_file", training_config,
                "--ltx2_checkpoint", ltx2_checkpoint,
            ]
            if latest_state_path:
                train_command.extend(["--resume", latest_state_path])

            self._run_stage_command(
                total_stages, total_stages, "Train LoRA", train_command, musubi_root, log_handle,
                [f"Output dir: {output_dir}",
                 f"Target steps this run: {completed_steps} -> {next_target_steps}",
                 f"Steps per run and save interval: {steps_per_run}",
                 f"Blocks to swap: {int(blocks_to_swap)}",
                 f"Learning rate: {effective_learning_rate}"],
            )

        latest_lora_path, latest_lora_step = self._latest_file(output_dir, output_name, ".safetensors")
        latest_comfy_lora_path, latest_comfy_step = self._latest_file(output_dir, output_name, ".comfy.safetensors")
        latest_state_path, latest_state_step = self._latest_state_dir(output_dir, output_name)

        completed_steps = max(latest_lora_step, latest_comfy_step, latest_state_step)
        if completed_steps < next_target_steps:
            raise RuntimeError(
                f"Training chunk did not produce the expected checkpoint. "
                f"Expected step {next_target_steps}, got {completed_steps}."
            )

        print(
            f"[LTXAVTools] post-run summary: state_step={latest_state_step} "
            f"lora_step={latest_lora_step} comfy_lora_step={latest_comfy_step}"
        )

        if keep_only_comfy_lora and latest_comfy_lora_path:
            deleted_count = self._delete_standard_lora_files(output_dir, output_name)
            latest_lora_path = ""
            print(f"[LTXAVTools] Deleted {deleted_count} standard LoRA file(s); keeping only Comfy LoRA files.")

        if copy_latest_to_comfy_loras:
            latest_comfy_lora_path = self._export_latest_to_comfy(latest_comfy_lora_path, output_name)

        applied_lora_path = latest_comfy_lora_path if latest_comfy_lora_path else latest_lora_path
        self._log_message(
            f"[LTXAVTools] Latest state path: {os.path.normpath(latest_state_path) if latest_state_path else '(none)'}",
            log_path,
        )
        self._log_message(
            f"[LTXAVTools] Latest standard LoRA: {os.path.normpath(latest_lora_path) if latest_lora_path else '(none)'}",
            log_path,
        )
        self._log_message(
            f"[LTXAVTools] Latest Comfy LoRA: {os.path.normpath(latest_comfy_lora_path) if latest_comfy_lora_path else '(none)'}",
            log_path,
        )
        self._log_message(
            f"[LTXAVTools] Applying LoRA to returned MODEL: "
            f"{os.path.normpath(applied_lora_path) if applied_lora_path else '(none)'} "
            f"with strength_model={float(strength_model)}",
            log_path,
        )
        output_model = self._apply_lora_to_model(model, applied_lora_path, strength_model)
        if applied_lora_path and os.path.isfile(applied_lora_path) and float(strength_model) != 0:
            self._log_message("[LTXAVTools] LoRA applied successfully to returned MODEL.", log_path)
        else:
            self._log_message("[LTXAVTools] Returned MODEL is unchanged (no LoRA file or strength_model=0).", log_path)

        return (
            output_model,
            os.path.normpath(latest_state_path) if latest_state_path else "",
            os.path.normpath(log_path),
            video_filename_prefix,
            output_name,
            int(completed_steps),
            int(total_target_steps),
        )


class LTXAV_CharacterLoraTraining(_LTXLoraTrainBase):
    DESCRIPTION = (
        "Runs the LTX-2.3 trainer with a character-LoRA preset using dynamic IMAGE and caption inputs."
    )
    MAX_IMAGE_SLOTS = 20
    PRESET_COPY_LATEST = False
    PRESET_KEEP_ONLY_COMFY = True

    RETURN_TYPES = ("MODEL", "STRING", "STRING", "STRING", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("model", "latest_state_path", "lora_path", "log_path", "video_filename_prefix", "output_name", "completed_steps", "total_target_steps")

    @classmethod
    def INPUT_TYPES(cls):
        optional_inputs = {
            f"image{i}": ("IMAGE", {"forceInput": True})
            for i in range(1, cls.MAX_IMAGE_SLOTS + 1)
        }
        optional_inputs.update({
            f"caption_{i}": ("STRING", {"default": "", "multiline": False})
            for i in range(1, cls.MAX_IMAGE_SLOTS + 1)
        })
        return {
            "required": {
                "model": ("MODEL",),
                "workspace_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Workspace folder for cache, output, logs, config, and the managed dynamic dataset.",
                }),
                "run_name": ("STRING", {
                    "default": "CharacterLoraTrainingRun",
                    "multiline": False,
                    "tooltip": "Run name used for logs.",
                }),
                "output_name": ("STRING", {
                    "default": "CharacterLoraTraining",
                    "multiline": False,
                    "tooltip": "LoRA output name used for checkpoints and downstream preview naming.",
                }),
                "image_count": ("INT", {
                    "default": 4, "min": 1, "max": cls.MAX_IMAGE_SLOTS, "step": 1,
                    "tooltip": "How many dynamic image inputs and caption fields to show.",
                }),
                "resolution_width": ("INT", {
                    "default": 1256, "min": 64, "max": 4096, "step": 8,
                    "tooltip": "Training bucket width.",
                }),
                "resolution_height": ("INT", {
                    "default": 1256, "min": 64, "max": 4096, "step": 8,
                    "tooltip": "Training bucket height.",
                }),
                "training_steps": ("INT", {
                    "default": 400, "min": 1, "max": 10000, "step": 1,
                    "tooltip": "Total number of training steps.",
                }),
                "num_repeats": ("INT", {
                    "default": 1, "min": 1, "max": 100, "step": 1,
                    "tooltip": "How many times each image is repeated per epoch.",
                }),
                "learning_rate": ("FLOAT", {
                    "default": 0.0002, "min": 1e-8, "max": 1.0, "step": 1e-6,
                    "tooltip": "Training learning rate.",
                }),
                "network_dim": ("INT", {
                    "default": 16, "min": 1, "max": 2048, "step": 1,
                    "tooltip": "LoRA rank. Higher values increase capacity but use more VRAM.",
                }),
                "network_alpha": ("INT", {
                    "default": 16, "min": 1, "max": 2048, "step": 1,
                    "tooltip": "LoRA alpha scaling. Typically set equal to or half of network_dim.",
                }),
                "lora_target_preset": (["full", "t2v", "v2v"], {
                    "default": "full",
                    "tooltip": "Which transformer layers get LoRA adapters. full trains all layers. t2v trains attention only. v2v adds feed-forward layers on top of t2v.",
                }),
                "blocks_to_swap": ("INT", {
                    "default": 0, "min": 0, "max": 64, "step": 1,
                    "tooltip": "How many transformer blocks to swap to CPU. 0 is fastest if VRAM allows.",
                }),
                "clear_memory_before_gemma": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clears Comfy and CUDA memory before the Gemma cache stage.",
                }),
                "gemma_recovery_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Experimental. Tries alternate Gemma cache settings after the normal path fails.",
                }),
                "gemma_load_in_4bit": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Loads Gemma in 4-bit mode during text encoder caching.",
                }),
                "cache_strategy": (["auto", "force", "skip"], {
                    "default": "auto",
                    "tooltip": "Cache behavior. auto reuses cache when present, force rebuilds, skip bypasses.",
                }),
                "strength_model": ("FLOAT", {
                    "default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                    "tooltip": "Strength when applying the newest trained LoRA back onto the returned MODEL.",
                }),
                "musubi_root": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Root folder of your musubi-tuner install.",
                }),
                "ltx2_checkpoint": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to the LTX-2.3 DiT checkpoint.",
                }),
                "gemma_root": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to the Gemma model root.",
                }),
            },
            "optional": optional_inputs,
        }

    def _extract_single_image_tensor(self, value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            tensor = value
            if tensor.ndim == 4:
                if int(tensor.shape[0]) <= 0:
                    return None
                return tensor[0]
            if tensor.ndim == 3:
                return tensor
            return None
        if isinstance(value, dict):
            for nested_value in value.values():
                tensor = self._extract_single_image_tensor(nested_value)
                if tensor is not None:
                    return tensor
            return None
        if isinstance(value, (list, tuple, set)):
            for nested_value in value:
                tensor = self._extract_single_image_tensor(nested_value)
                if tensor is not None:
                    return tensor
            return None
        return None

    def _save_dynamic_dataset_inputs(self, workspace_dir, image_count, kwargs):
        dataset_root = self._ensure_dir(os.path.join(workspace_dir, "dynamic_dataset"))
        images_dir = self._ensure_dir(os.path.join(dataset_root, "images"))

        for entry in os.scandir(images_dir):
            if not entry.is_file():
                continue
            ext = os.path.splitext(entry.name)[1].lower()
            if ext in self.IMAGE_EXTENSIONS or ext == ".txt":
                os.remove(entry.path)

        saved_count = 0
        for index in range(1, int(image_count) + 1):
            image_tensor = self._extract_single_image_tensor(kwargs.get(f"image{index}"))
            if image_tensor is None:
                continue
            image_array = image_tensor.detach().cpu().numpy()
            image_array = np.clip(image_array * 255.0, 0, 255).astype(np.uint8)
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            stem = f"image{index:03d}"
            image_path = os.path.join(images_dir, f"{stem}.png")
            caption_path = os.path.join(images_dir, f"{stem}.txt")
            cv2.imwrite(image_path, image_bgr)
            caption_text = str(kwargs.get(f"caption_{index}", "") or "").strip()
            with open(caption_path, "w", encoding="utf-8") as handle:
                handle.write(caption_text)
            saved_count += 1

        if saved_count <= 0:
            raise ValueError("No connected images were found. Connect at least one image input.")

        print(f"[LTXAVTools] Prepared dynamic dataset with {saved_count} image-caption pair(s): {images_dir}")
        return os.path.normpath(images_dir)

    def run(
        self,
        model,
        workspace_dir,
        run_name,
        output_name,
        image_count,
        resolution_width,
        resolution_height,
        training_steps,
        num_repeats,
        learning_rate,
        network_dim,
        network_alpha,
        lora_target_preset,
        blocks_to_swap,
        clear_memory_before_gemma,
        gemma_recovery_mode,
        gemma_load_in_4bit,
        cache_strategy,
        strength_model,
        musubi_root,
        ltx2_checkpoint,
        gemma_root,
        **kwargs,
    ):
        workspace_dir = self._norm(workspace_dir)
        managed_dataset_dir = self._save_dynamic_dataset_inputs(workspace_dir, image_count, kwargs)

        result = super().run(
            model=model,
            dataset_images_dir=managed_dataset_dir,
            workspace_dir=workspace_dir,
            run_name=run_name,
            output_name=output_name,
            resolution_width=resolution_width,
            resolution_height=resolution_height,
            steps_per_run=training_steps,
            total_target_steps=training_steps,
            network_dim=network_dim,
            network_alpha=network_alpha,
            blocks_to_swap=blocks_to_swap,
            clear_memory_before_gemma=clear_memory_before_gemma,
            gemma_recovery_mode=gemma_recovery_mode,
            gemma_load_in_4bit=gemma_load_in_4bit,
            learning_rate_preset="Custom",
            learning_rate=learning_rate,
            num_repeats=num_repeats,
            lora_target_preset=lora_target_preset,
            cache_strategy=cache_strategy,
            copy_latest_to_comfy_loras=self.PRESET_COPY_LATEST,
            keep_only_comfy_lora=self.PRESET_KEEP_ONLY_COMFY,
            strength_model=strength_model,
            create_captions=False,
            caption_text="",
            add_trigger_word=False,
            trigger_text="",
            musubi_root=musubi_root,
            ltx2_checkpoint=ltx2_checkpoint,
            gemma_root=gemma_root,
        )

        output_dir = os.path.join(workspace_dir, "output")
        lora_path, _ = self._latest_file(output_dir, self._safe_name(output_name, run_name), ".comfy.safetensors")
        if not lora_path:
            lora_path, _ = self._latest_file(output_dir, self._safe_name(output_name, run_name), ".safetensors")

        # insert lora_path after latest_state_path (index 1)
        return (result[0], result[1], os.path.normpath(lora_path) if lora_path else "", *result[2:])


class LTXAV_AudioLoraTraining(_LTXLoraTrainBase):
    DESCRIPTION = (
        "Continues character LoRA training using audio clips only, targeting the LTX-2.3 audio layers. "
        "Connect base_lora_path from LTXAV_CharacterLoraTraining to warm-start from the visual identity LoRA."
    )
    MAX_AUDIO_SLOTS = 20
    PRESET_COPY_LATEST = False
    PRESET_KEEP_ONLY_COMFY = True

    RETURN_TYPES = ("MODEL", "STRING", "STRING", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("model", "latest_state_path", "merged_lora_path", "log_path", "output_name", "completed_steps", "total_target_steps")

    @classmethod
    def INPUT_TYPES(cls):
        optional_inputs = {
            f"audio{i}": ("AUDIO", {"forceInput": True})
            for i in range(1, cls.MAX_AUDIO_SLOTS + 1)
        }
        optional_inputs.update({
            f"caption_{i}": ("STRING", {"default": "", "multiline": False})
            for i in range(1, cls.MAX_AUDIO_SLOTS + 1)
        })
        return {
            "required": {
                "model": ("MODEL",),
                "base_lora_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to the LoRA from Phase 1 character training. Used as --network_weights to warm-start audio layer training.",
                }),
                "workspace_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Working folder for audio cache, logs, config, and training state.",
                }),
                "run_name": ("STRING", {"default": "AudioLoraTrainingRun", "multiline": False}),
                "output_name": ("STRING", {"default": "AudioLoraTraining", "multiline": False}),
                "audio_count": ("INT", {
                    "default": 8, "min": 1, "max": cls.MAX_AUDIO_SLOTS, "step": 1,
                    "tooltip": "How many dynamic audio inputs and caption fields to show.",
                }),
                "training_steps": ("INT", {"default": 400, "min": 1, "max": 10000, "step": 1}),
                "num_repeats": ("INT", {"default": 2, "min": 1, "max": 100, "step": 1}),
                "learning_rate": ("FLOAT", {"default": 1e-4, "min": 1e-8, "max": 1.0, "step": 1e-6}),
                "network_dim": ("INT", {"default": 32, "min": 1, "max": 2048, "step": 1}),
                "network_alpha": ("INT", {"default": 16, "min": 1, "max": 2048, "step": 1}),
                "blocks_to_swap": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "audio_only_target_resolution": ("INT", {
                    "default": 64, "min": 32, "max": 4096, "step": 1,
                    "tooltip": "Square target resolution for audio-only latent geometry.",
                }),
                "audio_only_target_fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 240.0, "step": 0.1}),
                "audio_only_sequence_resolution": ("INT", {
                    "default": 64, "min": 0, "max": 8192, "step": 1,
                    "tooltip": "Virtual sequence resolution for shifted_logit_normal in audio mode.",
                }),
                "audio_bucket_strategy": (["pad", "truncate"], {"default": "pad"}),
                "audio_bucket_interval": ("FLOAT", {
                    "default": 2.0, "min": 0.1, "max": 120.0, "step": 0.1,
                    "tooltip": "Audio bucket step size in seconds.",
                }),
                "ltx2_audio_only_model": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Load the physically audio-only transformer variant.",
                }),
                "clear_memory_before_gemma": ("BOOLEAN", {"default": True}),
                "gemma_recovery_mode": ("BOOLEAN", {"default": False}),
                "gemma_load_in_4bit": ("BOOLEAN", {"default": False}),
                "cache_strategy": (["auto", "force", "skip"], {"default": "auto"}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "musubi_root": ("STRING", {"default": "", "multiline": False}),
                "ltx2_checkpoint": ("STRING", {"default": "", "multiline": False}),
                "gemma_root": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                **optional_inputs,
                "execution_signal": ("STRING", {"forceInput": True, "tooltip": "Connect any string output here to force this node to wait until that upstream node completes."}),
            },
        }

    # --- dataset helpers ---

    _AUDIO_LORA_KEY_FRAGMENTS = ("audio_attn", "video_to_audio_attn", "audio_ff")

    def _filter_lora_to_audio_keys(self, base_lora_path, dest_dir):
        from safetensors.torch import load_file, save_file
        state_dict = load_file(base_lora_path)
        filtered = {
            k: v for k, v in state_dict.items()
            if any(frag in k for frag in self._AUDIO_LORA_KEY_FRAGMENTS)
        }
        if not filtered:
            print("[LTXAVTools] base_lora_path has no audio keys — skipping network_weights warm-start")
            return None
        out_path = os.path.join(dest_dir, "_audio_filtered_base_lora.safetensors")
        save_file(filtered, out_path)
        print(f"[LTXAVTools] Filtered base LoRA to {len(filtered)} audio keys → {out_path}")
        return out_path

    def _merge_character_and_audio_loras(self, character_lora_path, audio_lora_path, output_dir, output_name):
        from safetensors.torch import load_file, save_file
        base_sd = load_file(character_lora_path)
        audio_sd = load_file(audio_lora_path)
        merged = {**base_sd, **audio_sd}  # audio keys overwrite character keys where they overlap
        out_path = os.path.join(output_dir, f"{output_name}-merged.comfy.safetensors")
        save_file(merged, out_path)
        print(f"[LTXAVTools] Merged LoRA: {len(base_sd)} base + {len(audio_sd)} audio = {len(merged)} keys → {out_path}")
        return os.path.normpath(out_path)

    def _extract_audio_waveform(self, value):
        if not isinstance(value, dict):
            return None, None
        waveform = value.get("waveform")
        sample_rate = value.get("sample_rate")
        if waveform is None or sample_rate is None:
            return None, None
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.as_tensor(waveform)
        if waveform.ndim == 3:
            waveform = waveform[0]
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        return waveform.float().detach().cpu(), int(sample_rate)

    def _save_dynamic_audio_dataset_inputs(self, workspace_dir, audio_count, kwargs):
        if torchaudio is None:
            raise ImportError("torchaudio is required for LTXAV_AudioLoraTraining.")

        dataset_root = self._ensure_dir(os.path.join(workspace_dir, "dynamic_dataset"))
        audio_dir = self._ensure_dir(os.path.join(dataset_root, "audio"))

        for entry in os.scandir(audio_dir):
            if entry.is_file():
                ext = os.path.splitext(entry.name)[1].lower()
                if ext in self.AUDIO_EXTENSIONS or ext == ".txt":
                    os.remove(entry.path)

        saved_count = 0
        for index in range(1, int(audio_count) + 1):
            audio_value = kwargs.get(f"audio{index}")
            waveform, sample_rate = self._extract_audio_waveform(audio_value)
            if waveform is None:
                continue
            stem = f"audio{index:03d}"
            audio_path = os.path.join(audio_dir, f"{stem}.wav")
            caption_path = os.path.join(audio_dir, f"{stem}.txt")
            torchaudio.save(audio_path, waveform, sample_rate)
            caption_text = str(kwargs.get(f"caption_{index}", "") or "").strip()
            with open(caption_path, "w", encoding="utf-8") as handle:
                handle.write(caption_text)
            saved_count += 1

        if saved_count <= 0:
            raise ValueError("No connected audio inputs found. Connect at least one AUDIO input.")

        print(f"[LTXAVTools] Prepared dynamic audio dataset with {saved_count} clip(s): {audio_dir}")
        return os.path.normpath(audio_dir)

    def _count_audio_dataset_files(self, audio_dir):
        audio_count = 0
        caption_count = 0
        if not os.path.isdir(audio_dir):
            return audio_count, caption_count
        for entry in os.scandir(audio_dir):
            if not entry.is_file():
                continue
            ext = os.path.splitext(entry.name)[1].lower()
            if ext in self.AUDIO_EXTENSIONS:
                audio_count += 1
            elif ext == ".txt":
                caption_count += 1
        return audio_count, caption_count

    # --- audio TOML writers ---

    def _write_audio_dataset_config(
        self,
        path,
        audio_directory,
        cache_dir,
        target_resolution,
        num_repeats,
        audio_bucket_strategy,
        audio_bucket_interval,
    ):
        content = (
            "[general]\n"
            f"resolution = [{int(target_resolution)}, {int(target_resolution)}]\n"
            'caption_extension = ".txt"\n'
            "batch_size = 1\n"
            "enable_bucket = true\n"
            "bucket_no_upscale = false\n\n"
            "[[datasets]]\n"
            f'audio_directory = "{self._quote(audio_directory)}"\n'
            f'cache_directory = "{self._quote(cache_dir)}"\n'
            f"num_repeats = {int(num_repeats)}\n"
            f'audio_bucket_strategy = "{audio_bucket_strategy}"\n'
            f"audio_bucket_interval = {float(audio_bucket_interval)}\n"
        )
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)

    def _write_audio_training_config(
        self,
        path,
        dataset_config,
        checkpoint,
        gemma_root,
        output_dir,
        log_dir,
        output_name,
        network_dim,
        network_alpha,
        blocks_to_swap,
        learning_rate,
        max_train_steps,
        steps_per_run,
        total_target_steps,
        audio_only_sequence_resolution,
        ltx2_audio_only_model,
        network_weights=None,
    ):
        checkpoint_name = os.path.basename(str(checkpoint or "")).lower()
        fp8_scaled = "fp8" not in checkpoint_name
        lines = [
            "# Auto-generated by LTXAVTools audio LoRA trainer",
            f"# total_target_steps = {int(total_target_steps)}",
            f"# chunk_target_steps_this_run = {int(max_train_steps)}",
            f"# save_interval_per_run = {int(steps_per_run)}",
            f'ltx2_checkpoint = "{self._quote(checkpoint)}"',
            f'gemma_root = "{self._quote(gemma_root)}"',
            f'dataset_config = "{self._quote(dataset_config)}"',
            "",
            'ltx_mode = "audio"',
            'ltx_version = "2.3"',
            'ltx_version_check_mode = "error"',
            'lora_target_preset = "audio"',
            f"ltx2_audio_only_model = {'true' if ltx2_audio_only_model else 'false'}",
            f"audio_only_sequence_resolution = {int(audio_only_sequence_resolution)}",
            "",
            "cache_text_encoder_outputs = true",
            "cache_text_encoder_outputs_to_disk = false",
            "",
            "fp8_base = true",
            f"fp8_scaled = {'true' if fp8_scaled else 'false'}",
            "sdpa = true",
            "gradient_checkpointing = true",
            "gradient_accumulation_steps = 1",
            f"blocks_to_swap = {int(blocks_to_swap)}",
            "",
            'optimizer_type = "AdamW8Bit"',
            f"learning_rate = {learning_rate}",
            'lr_scheduler = "constant_with_warmup"',
            "lr_warmup_steps = 100",
            "",
            'network_module = "networks.lora_ltx2"',
            f"network_dim = {int(network_dim)}",
            f"network_alpha = {int(network_alpha)}",
        ]
        if network_weights:
            lines.append(f'network_weights = "{self._quote(network_weights)}"')
        lines += [
            'timestep_sampling = "shifted_logit_normal"',
            "ltx2_first_frame_conditioning_p = 0.5",
            "",
            f'output_dir = "{self._quote(output_dir)}"',
            f'output_name = "{output_name}"',
            'log_with = "tensorboard"',
            f'logging_dir = "{self._quote(log_dir)}"',
            "log_config = true",
            f"max_train_steps = {int(max_train_steps)}",
            f"save_every_n_steps = {int(steps_per_run)}",
            'save_model_as = "safetensors"',
            'mixed_precision = "bf16"',
            "save_state = true",
            "save_state_on_train_end = true",
        ]
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    # --- run ---

    def run(
        self,
        model,
        base_lora_path,
        workspace_dir,
        run_name,
        output_name,
        audio_count,
        training_steps,
        num_repeats,
        learning_rate,
        network_dim,
        network_alpha,
        blocks_to_swap,
        audio_only_target_resolution,
        audio_only_target_fps,
        audio_only_sequence_resolution,
        audio_bucket_strategy,
        audio_bucket_interval,
        ltx2_audio_only_model,
        clear_memory_before_gemma,
        gemma_recovery_mode,
        gemma_load_in_4bit,
        cache_strategy,
        strength_model,
        musubi_root,
        ltx2_checkpoint,
        gemma_root,
        **kwargs,
    ):
        workspace_dir = self._norm(workspace_dir)
        musubi_root = self._norm(musubi_root)
        ltx2_checkpoint = self._norm(ltx2_checkpoint)
        gemma_root = self._norm(gemma_root)
        base_lora_path = self._norm(base_lora_path) if base_lora_path else ""

        run_name = self._safe_name(run_name, "AudioLoraTrainingRun")
        output_name = self._safe_name(output_name, run_name)
        audio_bucket_strategy = str(audio_bucket_strategy or "pad").strip().lower()

        workspace_dir = self._ensure_dir(workspace_dir)
        if not os.path.isdir(musubi_root):
            raise ValueError(f"musubi_root does not exist: {musubi_root}")
        if not os.path.isfile(ltx2_checkpoint):
            raise ValueError(f"ltx2_checkpoint does not exist: {ltx2_checkpoint}")
        if not os.path.isdir(gemma_root):
            raise ValueError(f"gemma_root does not exist: {gemma_root}")
        if base_lora_path and not os.path.isfile(base_lora_path):
            raise ValueError(f"base_lora_path does not exist: {base_lora_path}")

        dataset_audio_dir = self._save_dynamic_audio_dataset_inputs(workspace_dir, audio_count, kwargs)

        python_exe, accelerate_exe, env_source = self._resolve_musubi_executables(musubi_root)

        cache_dir = self._ensure_dir(os.path.join(workspace_dir, "audio_cache"))
        output_dir = self._ensure_dir(os.path.join(workspace_dir, "audio_output"))
        logs_dir = self._ensure_dir(os.path.join(workspace_dir, "audio_logs"))
        config_dir = self._ensure_dir(os.path.join(workspace_dir, "audio_config"))
        dataset_config = os.path.join(config_dir, "dataset-audio.toml")
        training_config = os.path.join(config_dir, "training_args_audio.toml")

        cache_signature_path = os.path.join(cache_dir, "ltxav_audio_cache_signature.json")
        current_cache_signature = {
            "dataset_audio_dir": dataset_audio_dir,
            "audio_only_target_resolution": int(audio_only_target_resolution),
            "audio_only_target_fps": float(audio_only_target_fps),
            "audio_only_sequence_resolution": int(audio_only_sequence_resolution),
            "audio_bucket_strategy": audio_bucket_strategy,
            "audio_bucket_interval": float(audio_bucket_interval),
            "num_repeats": int(num_repeats),
            "ltx2_audio_only_model": bool(ltx2_audio_only_model),
            "ltx2_mode": "audio",
        }

        latest_state_path, completed_steps = self._latest_state_dir(output_dir, output_name)
        if completed_steps >= int(training_steps):
            raise RuntimeError(
                f"Training complete: reached {completed_steps}/{int(training_steps)} steps. Stopping workflow."
            )

        network_weights_path = None
        if base_lora_path:
            network_weights_path = self._filter_lora_to_audio_keys(base_lora_path, config_dir)

        self._write_audio_dataset_config(
            dataset_config,
            dataset_audio_dir,
            cache_dir,
            audio_only_target_resolution,
            num_repeats,
            audio_bucket_strategy,
            audio_bucket_interval,
        )
        self._write_audio_training_config(
            training_config,
            dataset_config,
            ltx2_checkpoint,
            gemma_root,
            output_dir,
            logs_dir,
            output_name,
            network_dim,
            network_alpha,
            blocks_to_swap,
            learning_rate,
            training_steps,
            training_steps,
            training_steps,
            audio_only_sequence_resolution,
            ltx2_audio_only_model,
            network_weights=network_weights_path,
        )

        log_path = os.path.join(logs_dir, f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        with open(log_path, "w", encoding="utf-8") as log_handle:
            log_handle.write(f"dataset_audio_dir={dataset_audio_dir}\n")
            log_handle.write(f"workspace_dir={workspace_dir}\n")
            log_handle.write(f"base_lora_path={base_lora_path}\n")
            log_handle.write(f"completed_steps={completed_steps}\n")
            log_handle.write(f"training_steps={training_steps}\n\n")
            log_handle.flush()

            audio_count, caption_count = self._count_audio_dataset_files(dataset_audio_dir)

            cache_signature_matches = False
            if os.path.isfile(cache_signature_path):
                try:
                    with open(cache_signature_path, "r", encoding="utf-8") as handle:
                        cache_signature_matches = json.load(handle) == current_cache_signature
                except Exception:
                    pass

            if cache_strategy == "force":
                should_build_cache = True
            elif cache_strategy == "skip":
                should_build_cache = False
            else:
                has_latent = self._has_latent_cache_files(cache_dir)
                has_text = self._has_complete_text_encoder_cache_files(cache_dir)
                should_build_cache = not has_latent or not has_text or not cache_signature_matches

            total_stages = 3 if should_build_cache else 1

            print(f"[LTXAVTools] dataset_audio_dir={dataset_audio_dir}")
            print(f"[LTXAVTools] base_lora_path={base_lora_path or '(none)'}")
            print(f"[LTXAVTools] completed_steps={completed_steps} / {training_steps}")
            print(f"[LTXAVTools] musubi_python={python_exe}")
            print(f"[LTXAVTools] dataset summary: audio={audio_count} captions={caption_count}")

            gemma_load_in_8bit = not bool(gemma_load_in_4bit)

            if should_build_cache:
                self._run_stage_command(
                    1,
                    total_stages,
                    "Cache audio latents",
                    [
                        python_exe,
                        "ltx2_cache_latents.py",
                        "--dataset_config", dataset_config,
                        "--ltx2_checkpoint", ltx2_checkpoint,
                        "--device", "cuda",
                        "--vae_dtype", "bf16",
                        "--ltx2_mode", "audio",
                        "--audio_only_target_resolution", str(int(audio_only_target_resolution)),
                        "--audio_only_target_fps", str(float(audio_only_target_fps)),
                        "--audio_only_sequence_resolution", str(int(audio_only_sequence_resolution)),
                    ],
                    musubi_root,
                    log_handle,
                    [
                        f"Dataset audio dir: {dataset_audio_dir}",
                        f"Audio files: {audio_count}  Captions: {caption_count}",
                        f"Cache dir: {cache_dir}",
                    ],
                )
                if clear_memory_before_gemma:
                    self._clear_memory_before_gemma(log_handle)
                if gemma_recovery_mode:
                    self._run_text_encoder_cache_stage_with_recovery(
                        2, total_stages, "Cache audio text encoder outputs",
                        python_exe, dataset_config, ltx2_checkpoint, gemma_root,
                        "audio", gemma_load_in_8bit, gemma_recovery_mode,
                        musubi_root, log_handle,
                        [f"Gemma root: {gemma_root}"],
                        gemma_load_in_4bit=bool(gemma_load_in_4bit),
                    )
                else:
                    self._run_text_encoder_cache_stage(
                        2, total_stages, "Cache audio text encoder outputs",
                        python_exe, dataset_config, ltx2_checkpoint, gemma_root,
                        "audio", gemma_load_in_8bit,
                        musubi_root, log_handle,
                        [f"Gemma root: {gemma_root}"],
                        gemma_load_in_4bit=bool(gemma_load_in_4bit),
                    )
                with open(cache_signature_path, "w", encoding="utf-8") as handle:
                    json.dump(current_cache_signature, handle, indent=2, sort_keys=True)
            else:
                self._print_stage_banner(
                    log_handle, 1, total_stages, "Skip cache build",
                    [f"Cache strategy: {cache_strategy}", "Proceeding directly to training."],
                )

            train_command = [
                accelerate_exe, "launch",
                "--num_cpu_threads_per_process", "1",
                "--mixed_precision", "bf16",
                "ltx2_train_network.py",
                "--config_file", training_config,
                "--ltx2_checkpoint", ltx2_checkpoint,
                "--ltx2_mode", "audio",
                "--audio_only_sequence_resolution", str(int(audio_only_sequence_resolution)),
            ]
            if ltx2_audio_only_model:
                train_command.append("--ltx2_audio_only_model")
            if latest_state_path:
                train_command.extend(["--resume", latest_state_path])

            self._run_stage_command(
                total_stages, total_stages, "Train audio LoRA",
                train_command, musubi_root, log_handle,
                [
                    f"Output dir: {output_dir}",
                    f"Target steps: {completed_steps} -> {training_steps}",
                    f"base_lora_path (network_weights): {base_lora_path or '(none)'}",
                    f"blocks_to_swap: {int(blocks_to_swap)}",
                    f"learning_rate: {learning_rate}",
                ],
            )
            with open(cache_signature_path, "w", encoding="utf-8") as handle:
                json.dump(current_cache_signature, handle, indent=2, sort_keys=True)

        latest_lora_path, latest_lora_step = self._latest_file(output_dir, output_name, ".safetensors")
        latest_comfy_lora_path, latest_comfy_step = self._latest_file(output_dir, output_name, ".comfy.safetensors")
        latest_state_path, latest_state_step = self._latest_state_dir(output_dir, output_name)
        completed_steps = max(latest_lora_step, latest_comfy_step, latest_state_step)

        if completed_steps < training_steps:
            raise RuntimeError(
                f"Audio training did not reach the expected step. Expected {training_steps}, got {completed_steps}."
            )

        if self.PRESET_KEEP_ONLY_COMFY and latest_comfy_lora_path:
            self._delete_standard_lora_files(output_dir, output_name)
            latest_lora_path = ""

        if self.PRESET_COPY_LATEST and latest_comfy_lora_path:
            latest_comfy_lora_path = self._export_latest_to_comfy(latest_comfy_lora_path, output_name)

        applied_lora_path = latest_comfy_lora_path or latest_lora_path

        merged_lora_path = ""
        if base_lora_path and applied_lora_path:
            try:
                merged_lora_path = self._merge_character_and_audio_loras(
                    base_lora_path, applied_lora_path, output_dir, output_name
                )
            except Exception as exc:
                print(f"[LTXAVTools] WARNING: LoRA merge failed ({exc}); merged_lora_path will be empty")

        lora_to_apply = merged_lora_path if merged_lora_path else applied_lora_path
        output_model = self._apply_lora_to_model(model, lora_to_apply, strength_model)

        return (
            output_model,
            os.path.normpath(latest_state_path) if latest_state_path else "",
            merged_lora_path,
            os.path.normpath(log_path),
            output_name,
            int(completed_steps),
            int(training_steps),
        )


NODE_CLASS_MAPPINGS = {
    "LTXAV_CharacterLoraTraining": LTXAV_CharacterLoraTraining,
    "LTXAV_AudioLoraTraining": LTXAV_AudioLoraTraining,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXAV_CharacterLoraTraining": "LTXAV Character LoRA Training",
    "LTXAV_AudioLoraTraining": "LTXAV Audio LoRA Training",
}