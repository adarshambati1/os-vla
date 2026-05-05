# Operational-Space VLA (OS-VLA)

A Vision-Language-Action model that outputs **operational-space** commands
(end-effector pose + contact wrench + impedance — 24D total) instead of
joint-space targets, paired with Khatib's operational-space controller for
the 1 kHz inner loop.

## Why

Joint-space VLAs have to learn the inverse kinematics, the geometric
relationship to the environment, and the contact dynamics jointly — all
from pixels and language. Operational-space outputs hand the
geometry/dynamics off to a model-based controller and let the network
focus on the task-relevant abstractions: "where to put the tool", "how
hard to push", "how stiff to be against perturbations".

## Architecture

```
        image + instruction
                |
                v
       PaliGemma-3B (LoRA)
                |
            hidden state
                |
                v
       OperationalSpaceHead    --> [pose 6D, wrench 6D, impedance 12D]
                |                            |
                |                            v
                |                  Khatib OSC @ 1 kHz
                |                            |
                |                            v
                +-----<-----<-----<--- joint torques --> robot
```

Baseline: same backbone + `JointSpaceHead` outputting joint targets directly.

## Status

Scaffolding only. Code runs on CPU with dummy tensors; no Isaac Gym required
to develop. Real training / evaluation requires a Linux box with NVIDIA GPU
and Isaac Gym installed (see `docs/simtoolreal-notes.md` once written).

## Layout

```
configs/    YAML configs for training and data collection
data/       raw rollouts, processed dataset, build scripts
models/     action head, OS-VLA wrapper, Khatib controller
training/   training loop, LoRA config, losses
eval/       closed-loop eval, ablations, effect monitor
notebooks/  exploration and analysis notebooks
docs/       design notes, SimToolReal notes
tests/      CPU smoke tests
```

## Getting started (CPU dev)

```bash
# Create the dedicated env (Python 3.14)
conda env create -f environment.yml
conda activate osvla

# Or, if you already have a Python 3.10+ env you'd rather use:
pip install -r requirements.txt

# Smoke test — forward + loss on dummy tensors, no Isaac Gym needed.
python -m tests.smoke
```

## Getting started (GPU)

```bash
# 1. Install Isaac Gym (Linux + NVIDIA only, separate download from NVIDIA)
# 2. pip install -e .
# 3. python data/scripts/collect_rollouts.py --config configs/data_collection.yaml
# 4. python data/scripts/build_dataset.py
# 5. python training/train.py --config configs/train_baseline.yaml
# 6. python training/train.py --config configs/train_osvla.yaml
# 7. python eval/evaluate.py --ckpt runs/osvla/last.pt
```

## References

- SimToolReal (Kedia et al., 2026)
- PaliGemma — `google/paligemma-3b-pt-224`
- Khatib, "A unified approach for motion and force control of robot
  manipulators: The operational space formulation", 1987
- OpenVLA (Kim et al.), Dr.VLA (Swann et al.), ForceVLA (Xue et al.,
  2025), FAVLA (Li et al., 2026)
