"""LoRA hyperparameters for PaliGemma.

Defaults: r=16 on q/k/v/o projections, alpha=32, dropout=0.05.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LoRASpec:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )
    bias: str = "none"  # "none" | "all" | "lora_only"
    task_type: str = "CAUSAL_LM"

    def to_peft(self):
        from peft import LoraConfig

        return LoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=list(self.target_modules),
            bias=self.bias,
            task_type=self.task_type,
        )
