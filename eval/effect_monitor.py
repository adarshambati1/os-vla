"""Effect monitor: a closed-loop classifier over (image, instruction)
that predicts task progress / phase.

Decision (resolves Question 5 in the project notes):

We use the same PaliGemma backbone as the policy (with a separate LoRA
adapter and a small classification head) rather than a separate model.
Justification:

1. PaliGemma's vision tower is already well-aligned with manipulation
   imagery after policy fine-tuning, so the monitor inherits the
   representation cheaply.
2. Sharing the backbone halves the GPU memory at deploy time (LoRA
   adapters can be hot-swapped on the same base weights).
3. A separate, smaller monitor (e.g. ResNet+MLP) was considered but
   rejected because the monitor needs language conditioning to know
   what "done" means per task, and language alignment is what the VLM
   provides for free.

Phase labels (configurable per task):
    0  start / approaching
    1  contact established
    2  task in progress
    3  done / success
    4  failure / aborted

This module is the *training* side of the monitor — same dataset, same
backbone, separate adapter & head. The controller layer (when this is
wired into eval) reads ``argmax(phase_logits)`` and decides whether to
stop, persist, or re-plan.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


PHASE_LABELS = ("start", "contact", "in_progress", "done", "failure")


@dataclass
class EffectMonitorConfig:
    backbone_name: str = "google/paligemma-3b-pt-224"
    num_phases: int = 5
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05


class PhaseHead(nn.Module):
    def __init__(self, input_dim: int = 2048, num_phases: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_phases),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, targets)
