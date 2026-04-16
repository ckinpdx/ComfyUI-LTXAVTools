import numpy as np
import torch


class LTXSigmaResample:
    """
    Resamples a sigma schedule to a different step count via f(σ) ODE integration.

    Extracts the phase portrait f(σ) = dσ/dn from the input schedule, then
    integrates forward at the new step count. The output schedule preserves
    the essential character of the original — cliff positions, cluster density,
    tail shape — regardless of step count change.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "steps": ("INT", {
                    "default": 8, "min": 2, "max": 100, "step": 1,
                    "tooltip": "Number of steps for the resampled schedule.",
                }),
            }
        }

    RETURN_TYPES  = ("SIGMAS",)
    RETURN_NAMES  = ("sigmas",)
    FUNCTION      = "resample"
    CATEGORY      = "LTXAVTools/sampling"

    def resample(self, sigmas, steps):
        s = sigmas.numpy().astype(np.float64) if isinstance(sigmas, torch.Tensor) else np.array(sigmas, dtype=np.float64)

        # Extract f(σ) curve from source schedule
        src_N      = len(s) - 1
        sigma_pts  = s[:-1]
        d_sigma_dn = np.diff(s) * src_N   # dσ/dn in normalized step space

        # np.interp needs ascending x
        idx_asc       = np.argsort(sigma_pts)
        sigma_pts_asc = sigma_pts[idx_asc]
        f_pts_asc     = d_sigma_dn[idx_asc]

        # ODE integration at new step count
        N      = steps
        out    = np.zeros(N + 1)
        out[0] = 1.0

        for i in range(N):
            σ = out[i]
            if σ <= 0.0:
                break
            f_val    = float(np.interp(σ, sigma_pts_asc, f_pts_asc,
                                       left=f_pts_asc[0], right=f_pts_asc[-1]))
            out[i+1] = max(0.0, σ + f_val / N)

        out[-1] = 0.0

        print(f"[LTXSigmaResample] {src_N} → {N} steps | {out.round(4).tolist()}")
        return (torch.FloatTensor(out),)


NODE_CLASS_MAPPINGS = {
    "LTXSigmaResample": LTXSigmaResample,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXSigmaResample": "LTX Sigma Resample",
}
