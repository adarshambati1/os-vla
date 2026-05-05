"""Closed-loop evaluation in SimToolReal.

Skeleton: at each control step, the policy produces a 24D operational-
space command (or 29D joint target for the baseline); the OSC layer
turns the OS command into joint torques; the env steps; we tally task
success per the SimToolReal task-success criteria.

Run on a Linux/NVIDIA box with Isaac Gym. The arg ``--ckpt`` should
point at a checkpoint saved by ``training/train.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _require_isaacgym() -> None:
    try:
        import isaacgym  # noqa: F401
    except ImportError as e:
        sys.exit(f"Isaac Gym required: {e}")


def evaluate(ckpt_path: Path, num_episodes: int, task: str) -> dict:
    """Roll out the policy and report success rate + traj stats."""
    _require_isaacgym()
    raise NotImplementedError(
        "Wire up SimToolReal env + Khatib OSC + policy loading. "
        "See models/os_controller.py and docs/simtoolreal-notes.md."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--num-episodes", type=int, default=50)
    args = parser.parse_args()
    metrics = evaluate(args.ckpt, args.num_episodes, args.task)
    print(metrics)


if __name__ == "__main__":
    main()
