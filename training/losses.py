"""Loss functions for OS-VLA and the joint-space baseline.

OS loss is a weighted sum of MSE on the three components. Defaults are
all 1.0; override via config once we observe component magnitudes on
real data.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    pose: float = 1.0
    wrench: float = 1.0
    impedance: float = 1.0


def os_loss(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    weights: LossWeights = LossWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pose_l = F.mse_loss(pred["pose"], target["pose"])
    wrench_l = F.mse_loss(pred["wrench"], target["wrench"])
    imp_l = F.mse_loss(pred["impedance"], target["impedance"])
    total = (
        weights.pose * pose_l
        + weights.wrench * wrench_l
        + weights.impedance * imp_l
    )
    return total, {"pose": pose_l, "wrench": wrench_l, "impedance": imp_l}


def joint_loss(
    pred: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = F.mse_loss(pred, target)
    return loss, {"joint": loss}
