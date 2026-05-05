"""Ablation runner: 6D pose-only, 12D pose+wrench, 24D full OS command.

Implements ablations by masking the action head's output before the
controller. The model still predicts 24D — we just zero out the
components we want to ablate so we can use a single trained policy.

For zeroed wrench: feed-forward force becomes 0 (pure motion).
For zeroed impedance: fall back to a fixed default (config-specified)
since K=0 would make the controller open-loop.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class AblationSpec:
    name: str
    mask_pose: bool
    mask_wrench: bool
    mask_impedance: bool
    default_K: float = 200.0
    default_D: float = 20.0


PRESETS = {
    "pose_only": AblationSpec("pose_only", False, True, True),
    "pose_wrench": AblationSpec("pose_wrench", False, False, True),
    "full_24d": AblationSpec("full_24d", False, False, False),
}


def apply_ablation(pred: dict, spec: AblationSpec) -> dict:
    out = {k: v.clone() for k, v in pred.items()}
    if spec.mask_pose:
        out["pose"] = torch.zeros_like(out["pose"])
    if spec.mask_wrench:
        out["wrench"] = torch.zeros_like(out["wrench"])
    if spec.mask_impedance:
        n = out["impedance"].shape[-1] // 2
        defaults = torch.cat(
            [
                torch.full((n,), spec.default_K),
                torch.full((n,), spec.default_D),
            ]
        ).to(out["impedance"])
        out["impedance"] = defaults.expand_as(out["impedance"]).clone()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--num-episodes", type=int, default=50)
    args = parser.parse_args()

    from eval.evaluate import evaluate

    results = {}
    for name, spec in PRESETS.items():
        # In a full implementation, evaluate() takes an ablation hook;
        # for now we leave the integration as a TODO once evaluate()
        # itself is implemented.
        print(f"=== {name} === spec={spec}")
        results[name] = evaluate(args.ckpt, args.num_episodes, args.task)
    print(results)


if __name__ == "__main__":
    main()
