"""PaliGemma + LoRA + action-head wrapper.

The wrapper is intentionally lazy about the backbone: a CPU-only smoke
test can build the action head and a stub backbone (``DummyBackbone``)
without pulling down PaliGemma weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from models.action_head import build_action_head


@dataclass
class OSVLAConfig:
    backbone_name: str = "google/paligemma-3b-pt-224"
    output_space: str = "operational"  # "operational" | "joint"
    hidden_dim_head: int = 512
    joint_dim: int = 29  # 7 KUKA + 22 Sharpa hand (SimToolReal)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj", "k_proj", "o_proj")
    dtype: str = "float16"  # used for backbone load on GPU


class DummyBackbone(nn.Module):
    """Stand-in backbone for CPU smoke tests.

    Produces a deterministic hidden vector of the requested dimension so
    downstream code (action head, loss, optimizer) can be exercised
    without downloading PaliGemma.
    """

    def __init__(self, hidden_size: int = 2048) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.proj = nn.Linear(3, hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del attention_mask
        b = input_ids.shape[0]
        # Combine a few stats from inputs so gradients flow through.
        text_feat = input_ids.float().mean(dim=-1, keepdim=True)
        image_feat = pixel_values.float().mean(dim=(-1, -2, -3)).view(b, 1)
        bias = torch.ones((b, 1), dtype=image_feat.dtype, device=image_feat.device)
        feat = torch.cat([text_feat, image_feat, bias], dim=-1)
        return self.proj(feat)


class OSVLA(nn.Module):
    """Backbone + action head.

    ``backbone`` should expose a ``forward(input_ids, pixel_values, ...)``
    that returns a [B, hidden] tensor (last-token hidden state). Use
    :meth:`from_paligemma` to construct one wrapping the real PaliGemma
    backbone with LoRA applied.
    """

    def __init__(
        self,
        backbone: nn.Module,
        action_head: nn.Module,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.action_head = action_head
        self.hidden_size = hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        hidden = self.backbone(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )
        return self.action_head(hidden)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def dummy(cls, cfg: OSVLAConfig, hidden_size: int = 2048) -> "OSVLA":
        backbone = DummyBackbone(hidden_size=hidden_size)
        head = build_action_head(
            cfg.output_space,
            input_dim=hidden_size,
            hidden_dim=cfg.hidden_dim_head,
            joint_dim=cfg.joint_dim,
        )
        return cls(backbone, head, hidden_size)

    @classmethod
    def from_paligemma(cls, cfg: OSVLAConfig) -> "OSVLA":
        """Build the real PaliGemma + LoRA + head stack.

        Imports ``transformers`` and ``peft`` lazily so that CPU smoke
        tests don't require either to be installed.
        """
        from peft import LoraConfig, get_peft_model
        from transformers import PaliGemmaForConditionalGeneration

        dtype = getattr(torch, cfg.dtype)
        base = PaliGemmaForConditionalGeneration.from_pretrained(
            cfg.backbone_name, torch_dtype=dtype
        )
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=list(cfg.target_modules),
            lora_dropout=cfg.lora_dropout,
            task_type="CAUSAL_LM",
        )
        base = get_peft_model(base, lora_cfg)

        hidden_size = base.config.text_config.hidden_size

        class _Wrap(nn.Module):
            def __init__(self, model: nn.Module) -> None:
                super().__init__()
                self.model = model

            def forward(self, input_ids, pixel_values, attention_mask=None):
                outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                # Last-token hidden state of the language tower.
                last = outputs.hidden_states[-1]
                if attention_mask is not None:
                    # Pick the actual last non-pad token.
                    idx = attention_mask.sum(dim=-1) - 1
                    batch_idx = torch.arange(last.shape[0], device=last.device)
                    return last[batch_idx, idx]
                return last[:, -1, :]

        backbone = _Wrap(base)
        head = build_action_head(
            cfg.output_space,
            input_dim=hidden_size,
            hidden_dim=cfg.hidden_dim_head,
            joint_dim=cfg.joint_dim,
        )
        return cls(backbone, head, hidden_size)
