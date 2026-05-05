"""Combine raw rollouts + per-step OS labels + camera images +
instructions into a Hugging Face ``datasets.Dataset`` ready for
training.

Per-sample schema:

    image           PIL.Image (RGB, 224x224 — PaliGemma-pt-224)
    instruction     str
    pose_target     float32 (6,)
    wrench_target   float32 (6,)
    impedance_target float32 (12,)
    joint_target    float32 (29,)
    task            str
    episode_id      str
    t               int

Dataset is saved with ``Dataset.save_to_disk(out_dir)``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from data.scripts.extract_os_labels import OSLabels, extract_labels


# Per-(category, task) instruction templates. Concrete strings; we'll
# diversify with a paraphrase pass once we have a base dataset.
TASK_INSTRUCTIONS: dict[str, str] = {
    "hammer_swing_down": "Pick up the hammer and swing it down onto the target.",
    "hammer_swing_side": "Pick up the hammer and swing it sideways toward the target.",
    "marker_draw_smile": "Use the marker to draw a smile on the surface.",
    "marker_write_c": "Use the marker to write the letter C.",
    "eraser_wipe_smile": "Use the eraser to wipe away the smile.",
    "eraser_wipe_c": "Use the eraser to wipe away the letter C.",
    "brush_sweep_forward": "Sweep the debris forward with the brush.",
    "brush_sweep_right": "Sweep the debris to the right with the brush.",
    "spatula_serve_plate": "Use the spatula to serve the food onto the plate.",
    "spatula_flip_over": "Use the spatula to flip the object over.",
    "screwdriver_spin_vertical": "Use the screwdriver vertically.",
    "screwdriver_spin_horizontal": "Use the screwdriver horizontally.",
}


def _maybe_load_image(rec: dict, t: int, default_size: int = 224) -> Image.Image:
    """Return the camera frame for step ``t`` or a placeholder.

    SimToolReal recordings don't include images by default; once we run
    the camera-attached collection script the field will be
    ``camera_rgb_array`` of shape ``(T, H, W, 3)``.
    """
    if "camera_rgb_array" in rec:
        frame = rec["camera_rgb_array"][t]
        return Image.fromarray(frame).resize((default_size, default_size))
    placeholder = np.zeros((default_size, default_size, 3), dtype=np.uint8)
    return Image.fromarray(placeholder)


def _episode_samples(
    npz_path: Path,
    task_name: str,
    stride: int = 1,
    image_size: int = 224,
) -> Iterator[dict]:
    with np.load(npz_path, allow_pickle=True) as data:
        rec = {k: data[k] for k in data.files}
    labels = extract_labels(npz_path)
    instruction = TASK_INSTRUCTIONS.get(
        task_name, f"Perform task: {task_name.replace('_', ' ')}."
    )
    episode_id = npz_path.stem
    T = labels.pose.shape[0]
    for t in range(0, T, stride):
        yield {
            "image": _maybe_load_image(rec, t, default_size=image_size),
            "instruction": instruction,
            "pose_target": labels.pose[t],
            "wrench_target": labels.wrench[t],
            "impedance_target": labels.impedance[t],
            "joint_target": labels.joint_pos[t],
            "task": task_name,
            "episode_id": episode_id,
            "t": int(t),
        }


def build(raw_root: Path, out_dir: Path, stride: int = 1) -> None:
    """Walk ``raw_root/<task>/<episode>.npz`` and assemble the dataset."""
    from datasets import Dataset, Features, Image as HFImage, Sequence, Value

    features = Features(
        {
            "image": HFImage(),
            "instruction": Value("string"),
            "pose_target": Sequence(Value("float32"), length=6),
            "wrench_target": Sequence(Value("float32"), length=6),
            "impedance_target": Sequence(Value("float32"), length=12),
            "joint_target": Sequence(Value("float32"), length=29),
            "task": Value("string"),
            "episode_id": Value("string"),
            "t": Value("int32"),
        }
    )

    def gen():
        for task_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
            task_name = task_dir.name
            for npz_path in sorted(task_dir.glob("*.npz")):
                yield from _episode_samples(npz_path, task_name, stride=stride)

    ds = Dataset.from_generator(gen, features=features)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    print(f"wrote dataset with {len(ds)} samples to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    build(args.raw, args.out, stride=args.stride)


if __name__ == "__main__":
    main()
