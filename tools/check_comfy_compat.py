"""
Check whether a ComfyUI update has broken (or is about to break) this pack.

Two halves:

  UPSTREAM  — fetches (never pulls) and lists incoming commits that touch the
              core files this pack reaches into. A commit elsewhere in ComfyUI
              is not interesting; one in comfy/samplers.py is.

  INSTALLED — imports the core APIs the pack depends on and asserts the exact
              properties it relies on, not merely that the names exist. The
              guider fix depends on `inner_set_conds` ASSIGNING into
              `self.original_conds` (that aliasing is what the fix works
              around); the schedule windowing depends on SplitSigmas SHARING
              the boundary sigma. Presence checks would pass while both
              silently changed meaning.

Exit codes: 0 = clean, 1 = a check failed, 2 = could not run.
Never modifies the working tree. `git fetch` only.

    python tools/check_comfy_compat.py [--comfy C:\\ComfyUI] [--quiet]
"""

import argparse
import os
import subprocess
import sys
import traceback

# Core paths whose changes can plausibly break this pack.
WATCHED = [
    "comfy/samplers.py",
    "comfy/sampler_helpers.py",
    "comfy/model_patcher.py",
    "comfy/nested_tensor.py",
    "comfy/ldm/lightricks/",
    "comfy_extras/nodes_lt.py",
    "comfy_extras/nodes_lt_audio.py",
    "comfy_extras/nodes_custom_sampler.py",
    "comfy_extras/nodes_hunyuan.py",   # LatentUpscaleModelLoader lives here
    "node_helpers.py",
]


def git(comfy, *args):
    return subprocess.run(["git", "-C", comfy, *args],
                          capture_output=True, text=True, timeout=180)


def check_upstream(comfy, log):
    log("=" * 72)
    log("UPSTREAM")
    log("=" * 72)
    head = git(comfy, "rev-parse", "--short", "HEAD").stdout.strip()
    desc = git(comfy, "describe", "--tags").stdout.strip()
    log(f"  local HEAD : {head}  ({desc})")

    f = git(comfy, "fetch", "--tags", "--quiet")
    if f.returncode != 0:
        log(f"  ! fetch failed: {f.stderr.strip()[:200]}")
        return None

    branch = git(comfy, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    upstream = git(comfy, "rev-parse", "--abbrev-ref", "@{u}").stdout.strip() \
        or f"origin/{branch}"
    behind = git(comfy, "rev-list", "--count", f"HEAD..{upstream}").stdout.strip()
    if not behind.isdigit():
        log(f"  ! could not compare against {upstream}")
        return None
    n = int(behind)
    log(f"  upstream   : {upstream}  ({n} commit(s) ahead of you)")
    if n == 0:
        log("  -> up to date")
        return 0

    hits = git(comfy, "log", "--oneline", f"HEAD..{upstream}", "--", *WATCHED).stdout.strip()
    if hits:
        lines = hits.splitlines()
        log(f"\n  !! {len(lines)} incoming commit(s) touch files this pack depends on:")
        for line in lines[:25]:
            log(f"       {line}")
        if len(lines) > 25:
            log(f"       ... and {len(lines) - 25} more")
        log("\n  Files affected:")
        names = git(comfy, "diff", "--name-only", f"HEAD..{upstream}", "--", *WATCHED).stdout
        for nm in sorted(set(names.split())):
            log(f"       {nm}")
    else:
        log(f"  -> {n} incoming commit(s), none touching watched files")
    return len(hits.splitlines()) if hits else 0


def check_installed(comfy, log):
    log("\n" + "=" * 72)
    log("INSTALLED API")
    log("=" * 72)
    sys.path.insert(0, comfy)
    os.chdir(comfy)
    results = []

    def check(name, fn, why):
        try:
            passed = bool(fn())
        except Exception as e:
            passed, why = False, f"{why} -- RAISED {type(e).__name__}: {e}"
        results.append((passed, name, why))

    import torch
    import inspect
    import comfy.samplers as S
    import comfy.sampler_helpers as SH
    import node_helpers

    # The shallow-copy guider fix (v1.6.1) only makes sense while set_conds
    # assigns into self.original_conds. If that ever stops being true, the fix
    # is either unnecessary or wrong -- either way it needs revisiting.
    check("CFGGuider.inner_set_conds assigns into original_conds",
          lambda: "self.original_conds[k]" in inspect.getsource(S.CFGGuider.inner_set_conds),
          "v1.6.1 shallow-copy fix depends on this aliasing")

    # Schedule windowing (v1.9.0) mirrors this slicing exactly.
    def split_ok():
        from comfy_extras.nodes_custom_sampler import SplitSigmas
        sg = torch.linspace(1, 0, 13)
        hi, lo = SplitSigmas().get_sigmas(sg, 4)
        return len(hi) == 5 and len(lo) == 9 and float(hi[-1]) == float(lo[0])
    check("SplitSigmas shares the boundary sigma", split_ok,
          "sampling_start/end_step windowing mirrors this")

    # ref_audio_bank / ic_reference_pack ride conditioning as custom keys.
    def cond_ok():
        c = node_helpers.conditioning_set_values(
            [[torch.randn(1, 4, 8), {}]], {"ref_audio_bank": {1: 1}})
        return "ref_audio_bank" in SH.convert_cond(c)[0]
    check("convert_cond preserves custom conditioning keys", cond_ok,
          "ref_audio_bank and ic_reference_pack ride here")

    check("linear_quadratic_schedule present",
          lambda: hasattr(S, "linear_quadratic_schedule"), "scheduler docs reference it")
    check("beta_scheduler present", lambda: hasattr(S, "beta_scheduler"), "")

    def nested_ok():
        from comfy.nested_tensor import NestedTensor
        v = torch.randn(1, 8, 6, 4, 4); a = torch.randn(1, 4, 20, 8)
        nt = NestedTensor([v, a])
        # The AV-latent guard depends on .shape reporting the VIDEO shape and
        # slicing hitting BOTH tensors -- that combination is what makes a
        # mis-wired AV latent silent rather than loud.
        return nt.shape[2] == 6 and nt[:, :, 0:3].tensors[1].shape[2] == 3
    check("NestedTensor shape/slice semantics unchanged", nested_ok,
          "_reject_av_latent guard depends on this")

    import comfy.ldm.lightricks.av_model as AV
    avsrc = inspect.getsource(AV)
    for sym in ["audio_to_video_attn", "video_to_audio_attn", "a2v_cross_attn",
                "v2a_cross_attn", "scale_shift_table_a2v_ca_video", "get_ada_values"]:
        check(f"av_model: {sym}", lambda s=sym: s in avsrc, "")
    check("av_model: a2v gate addcmul_ hook point",
          lambda: "addcmul_(a2v_out" in avsrc, "cross-attn amplification hook")

    from comfy_extras.nodes_lt import LTXVAddGuide
    for m in ["execute", "append_keyframe", "dilate_latent", "replace_latent_frames"]:
        check(f"LTXVAddGuide.{m}", lambda mm=m: hasattr(LTXVAddGuide, mm), "")

    # v1.9.1: the loader returns a CoreModelPatcher, not a bare nn.Module.
    def loader_shape():
        import comfy_extras.nodes_hunyuan as NH
        src = inspect.getsource(NH.LatentUpscaleModelLoader)
        return ("ModelPatcher" in src, "returns a patcher" if "ModelPatcher" in src
                else "returns a bare module")
    try:
        patched, note = loader_shape()
        log(f"\n  note: LatentUpscaleModelLoader {note} "
            f"-- upsampler shim handles both\n")
    except Exception:
        pass

    width = max(len(r[1]) for r in results)
    failed = 0
    for passed, name, why in results:
        if not passed:
            failed += 1
        log(f"  [{'OK  ' if passed else 'FAIL'}] {name:<{width}}  {why}")
    log(f"\n  {len(results) - failed}/{len(results)} passed")
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default=r"C:\ComfyUI")
    ap.add_argument("--quiet", action="store_true",
                    help="only print when something needs attention")
    a = ap.parse_args()

    buf = []
    log = buf.append
    try:
        incoming = check_upstream(a.comfy, log)
        failed = check_installed(a.comfy, log)
    except Exception:
        print("check_comfy_compat: could not run\n" + traceback.format_exc())
        return 2

    needs_attention = bool(failed) or bool(incoming)
    if not a.quiet or needs_attention:
        print("\n".join(buf))
    if needs_attention:
        print("\n>>> ATTENTION: "
              + (f"{failed} installed-API check(s) FAILED. " if failed else "")
              + (f"{incoming} incoming commit(s) touch watched files. " if incoming else "")
              + "Review before updating or generating.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
