"""Run the SimToolReal Isaac Gym envs and dump rollouts to disk.

Skeleton only. Real execution requires:
- Linux with NVIDIA GPU
- Isaac Gym installed (separate download from NVIDIA)
- SimToolReal repo cloned and on PYTHONPATH (we keep it as a sibling
  directory: ../simtoolreal-reference)

The script wraps a SimToolReal task, runs the pretrained policy (or any
policy implementing ``act(obs)``), and saves a ``RecordedData``-style
``.npz`` per episode under ``data/raw/<task_name>/<episode_id>.npz``.

Run:
    python data/scripts/collect_rollouts.py --config configs/data_collection.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def _require_isaacgym() -> None:
    """Fail loudly if Isaac Gym (Linux/NVIDIA) is not available."""
    try:
        import isaacgym  # noqa: F401
    except ImportError as e:
        sys.exit(
            "Isaac Gym is not installed. "
            "This script only runs on Linux + NVIDIA GPU with Isaac Gym. "
            f"Original error: {e}"
        )


def collect(cfg: dict) -> None:
    """Run rollouts and persist them.

    Pseudocode (filled in once the dependency stack lands):

        env = make_simtoolreal_env(
            task=cfg["task"],
            num_envs=cfg["num_envs"],
            with_fingertip_force_sensors=True,  # for wrench labels
        )
        policy = load_policy(cfg["policy_ckpt"])
        for episode in range(cfg["num_episodes"]):
            obs = env.reset()
            traj = []
            for t in range(cfg["episode_length"]):
                action = policy.act(obs)
                obs, reward, done, info = env.step(action)
                traj.append(snapshot(env, obs, action))
                if done.all():
                    break
            save_npz(traj, out_dir / f"{episode:06d}.npz")
    """
    _require_isaacgym()
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(
        "Wire up SimToolReal env construction and policy load — "
        "see docs/simtoolreal-notes.md for entry points."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    with args.config.open() as f:
        cfg = yaml.safe_load(f)
    collect(cfg)


if __name__ == "__main__":
    main()
