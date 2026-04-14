import numpy as np
import torch


class LTXDetailSigmas:
    """
    Generates a sigma schedule following the 'high-cluster + cliff + power-tail' principle.

    Divides steps into three regions:
      1. High cluster  — linear steps near sigma=1.0 for fine structure establishment
      2. Cliff         — single large step to cliff_sigma (distilled model shortcut)
      3. Tail          — power curve from cliff_sigma down to 0.0

    At 8 steps with defaults this approximates the known community schedule:
    1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.65, ~0.289, ~0.072, 0.0
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {
                    "default": 8, "min": 3, "max": 100, "step": 1,
                    "tooltip": "Total number of denoising steps.",
                }),
                "high_fraction": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 0.8, "step": 0.05,
                    "tooltip": "Fraction of steps spent in the high-sigma cluster.",
                }),
                "cluster_width": ("FLOAT", {
                    "default": 0.025, "min": 0.005, "max": 0.1, "step": 0.005,
                    "tooltip": "Sigma range covered by the high cluster (1.0 down to 1.0 - cluster_width).",
                }),
                "cliff_sigma": ("FLOAT", {
                    "default": 0.65, "min": 0.1, "max": 0.95, "step": 0.01,
                    "tooltip": "Target sigma for the cliff step (distilled shortcut landing point).",
                }),
                "tail_power": ("FLOAT", {
                    "default": 2.0, "min": 0.5, "max": 6.0, "step": 0.25,
                    "tooltip": "Power curve exponent for tail distribution. Higher = more steps near cliff_sigma.",
                }),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "get_sigmas"
    CATEGORY = "LTXAVTools/sampling"

    def get_sigmas(self, steps, high_fraction, cluster_width, cliff_sigma, tail_power):
        n_high = max(1, round(steps * high_fraction))
        n_tail = max(1, steps - n_high - 1)  # 1 step reserved for cliff

        cluster_bottom = 1.0 - cluster_width

        # High cluster: linear from 1.0 down to cluster_bottom
        high = np.linspace(1.0, cluster_bottom, n_high + 1)

        # Tail: power curve from cliff_sigma to 0.0 (skip t=0 since cliff_sigma is boundary)
        t = np.linspace(0.0, 1.0, n_tail + 1)[1:]
        tail = cliff_sigma * (1.0 - t) ** tail_power
        tail[-1] = 0.0

        sigmas = np.concatenate([high, [cliff_sigma], tail])

        print(f"[LTXDetailSigmas] {steps} steps → {sigmas.round(4).tolist()}")

        return (torch.FloatTensor(sigmas),)


NODE_CLASS_MAPPINGS = {
    "LTXDetailSigmas": LTXDetailSigmas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXDetailSigmas": "LTX Detail Sigmas",
}
