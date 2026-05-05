"""OS-VLA training loop.

Runs on GPU against the processed Hugging Face dataset built by
``data/scripts/build_dataset.py``. The CPU smoke test in
``tests/smoke.py`` covers the same code-paths against ``OSVLA.dummy``.

Usage:
    python training/train.py --config configs/train_osvla.yaml
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from models.osvla import OSVLA, OSVLAConfig
from training.losses import LossWeights, joint_loss, os_loss


@dataclass
class TrainConfig:
    output_space: str = "operational"  # "operational" | "joint"
    backbone_name: str = "google/paligemma-3b-pt-224"
    dataset_dir: str = "data/processed"
    output_dir: str = "runs/osvla"
    batch_size: int = 16
    grad_accum: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    num_epochs: int = 3
    log_every: int = 25
    save_every: int = 500
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {"pose": 1.0, "wrench": 1.0, "impedance": 1.0}
    )
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    joint_dim: int = 29
    use_wandb: bool = False
    wandb_project: str = "os-vla"
    seed: int = 0
    dtype: str = "bfloat16"


def _load_cfg(path: Path) -> TrainConfig:
    with path.open() as f:
        raw = yaml.safe_load(f)
    return TrainConfig(**raw)


def _collate(processor, batch: list[dict[str, Any]], output_space: str):
    instructions = [b["instruction"] for b in batch]
    images = [b["image"] for b in batch]
    encoded = processor(text=instructions, images=images, return_tensors="pt", padding=True)
    out = {
        "input_ids": encoded["input_ids"],
        "pixel_values": encoded["pixel_values"],
        "attention_mask": encoded.get("attention_mask"),
    }
    if output_space == "operational":
        out["pose_target"] = torch.tensor(
            [b["pose_target"] for b in batch], dtype=torch.float32
        )
        out["wrench_target"] = torch.tensor(
            [b["wrench_target"] for b in batch], dtype=torch.float32
        )
        out["impedance_target"] = torch.tensor(
            [b["impedance_target"] for b in batch], dtype=torch.float32
        )
    else:
        out["joint_target"] = torch.tensor(
            [b["joint_target"] for b in batch], dtype=torch.float32
        )
    return out


def _step(model: OSVLA, batch: dict, output_space: str, weights: LossWeights):
    pred = model(
        input_ids=batch["input_ids"],
        pixel_values=batch["pixel_values"],
        attention_mask=batch.get("attention_mask"),
    )
    if output_space == "operational":
        targets = {
            "pose": batch["pose_target"],
            "wrench": batch["wrench_target"],
            "impedance": batch["impedance_target"],
        }
        return os_loss(pred, targets, weights)
    return joint_loss(pred, batch["joint_target"])


def train(cfg: TrainConfig) -> None:
    from datasets import load_from_disk
    from transformers import PaliGemmaProcessor

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = OSVLAConfig(
        backbone_name=cfg.backbone_name,
        output_space=cfg.output_space,
        joint_dim=cfg.joint_dim,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        dtype=cfg.dtype,
    )
    model = OSVLA.from_paligemma(model_cfg).to(device)

    processor = PaliGemmaProcessor.from_pretrained(cfg.backbone_name)
    dataset = load_from_disk(cfg.dataset_dir)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=lambda b: _collate(processor, b, cfg.output_space),
    )

    weights = LossWeights(**cfg.loss_weights)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.use_wandb:
        import wandb

        wandb.init(project=cfg.wandb_project, config=cfg.__dict__)

    step = 0
    for epoch in range(cfg.num_epochs):
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            loss, parts = _step(model, batch, cfg.output_space, weights)
            (loss / cfg.grad_accum).backward()

            if (step + 1) % cfg.grad_accum == 0:
                optim.step()
                optim.zero_grad(set_to_none=True)

            if step % cfg.log_every == 0:
                detail = " ".join(f"{k}={v.item():.4f}" for k, v in parts.items())
                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"epoch={epoch} step={step} loss={loss.item():.4f} {detail}"
                )
                if cfg.use_wandb:
                    import wandb

                    wandb.log({"loss": loss.item(), **{k: v.item() for k, v in parts.items()}}, step=step)

            if step > 0 and step % cfg.save_every == 0:
                ckpt = out_dir / f"step_{step}.pt"
                torch.save(
                    {
                        "step": step,
                        "model_state": {
                            k: v
                            for k, v in model.state_dict().items()
                            if "lora" in k or "action_head" in k
                        },
                        "cfg": cfg.__dict__,
                    },
                    ckpt,
                )
                print(f"saved {ckpt}")

            step += 1

    final = out_dir / "last.pt"
    torch.save(
        {"step": step, "model_state": model.state_dict(), "cfg": cfg.__dict__},
        final,
    )
    print(f"saved {final}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = _load_cfg(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
