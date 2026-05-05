"""Action heads attached to the PaliGemma backbone.

Two variants share the same MLP shape so the only difference between the
OS-VLA and joint-space baseline is the output representation:

- ``OperationalSpaceHead`` -> 24D (pose 6 + wrench 6 + impedance 12)
- ``JointSpaceHead``       -> joint-dim (e.g. 16 for the Allegro hand)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


POSE_DIM = 6
WRENCH_DIM = 6
IMPEDANCE_DIM = 12  # 6 stiffness diag + 6 damping diag
OS_DIM = POSE_DIM + WRENCH_DIM + IMPEDANCE_DIM  # 24


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OperationalSpaceHead(nn.Module):
    """Predict a 24D operational-space command from a hidden state.

    Stiffness and damping are constrained positive via softplus; an
    optional learnable lower bound keeps the controller from going to a
    fully compliant (zero stiffness) regime.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        impedance_floor: float = 1.0,
    ) -> None:
        super().__init__()
        self.mlp = _MLP(input_dim, hidden_dim, OS_DIM)
        self.impedance_floor = impedance_floor

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.mlp(x)
        pose = out[..., :POSE_DIM]
        wrench = out[..., POSE_DIM : POSE_DIM + WRENCH_DIM]
        raw_imp = out[..., POSE_DIM + WRENCH_DIM :]
        impedance = F.softplus(raw_imp) + self.impedance_floor
        return {"pose": pose, "wrench": wrench, "impedance": impedance}

    def flatten(self, pred: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([pred["pose"], pred["wrench"], pred["impedance"]], dim=-1)


class JointSpaceHead(nn.Module):
    """Baseline: same backbone, joint-position output.

    Default ``joint_dim=29`` matches the SimToolReal embodiment
    (7-DOF KUKA arm + 22-DOF Sharpa hand). Override via config for
    other embodiments.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        joint_dim: int = 29,
    ) -> None:
        super().__init__()
        self.joint_dim = joint_dim
        self.mlp = _MLP(input_dim, hidden_dim, joint_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def build_action_head(
    output_space: str,
    input_dim: int = 2048,
    hidden_dim: int = 512,
    joint_dim: int = 29,
) -> nn.Module:
    if output_space == "operational":
        return OperationalSpaceHead(input_dim=input_dim, hidden_dim=hidden_dim)
    if output_space == "joint":
        return JointSpaceHead(input_dim=input_dim, hidden_dim=hidden_dim, joint_dim=joint_dim)
    raise ValueError(f"unknown output_space {output_space!r}")
