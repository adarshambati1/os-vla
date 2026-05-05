"""Extract operational-space labels from a SimToolReal recording.

Input: a ``RecordedData``-style ``.npz`` (see docs/simtoolreal-notes.md).
Output: per-timestep labels usable for training the OS-VLA head:

    pose       (T, 6)   palm-center pos (3) + axis-angle (3)
    wrench     (T, 6)   forces (3) + torques (3) at the tool tip
    impedance  (T, 12)  6 stiffness + 6 damping diagonals
    joint_pos  (T, 29)  joint positions (for the baseline head)

This module imports only numpy so it runs anywhere — no Isaac Gym
dependency. The wrench extraction expects the recording to contain
either (a) a per-fingertip force-sensor field, or (b) a tool-tip
force-sensor field. If neither is present we currently fill zeros and
emit a warning, so we can still iterate on shapes/code-paths during
development.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class OSLabels:
    pose: np.ndarray         # (T, 6)
    wrench: np.ndarray       # (T, 6)
    impedance: np.ndarray    # (T, 12)
    joint_pos: np.ndarray    # (T, 29)
    timestamps: np.ndarray   # (T,)


def quat_xyzw_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Vectorized xyzw quaternion -> axis-angle (3D)."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-8)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    norm_xyz = np.sqrt(x * x + y * y + z * z).clip(min=1e-8)
    angle = 2.0 * np.arctan2(norm_xyz, np.abs(w))
    sign = np.where(w >= 0, 1.0, -1.0)
    axis = np.stack([x, y, z], axis=-1) / norm_xyz[..., None]
    return axis * (angle * sign)[..., None]


def _extract_pose(rec: dict) -> np.ndarray:
    """Palm-center pose (T, 6) from robot root state.

    SimToolReal's ``robot_root_states_array`` is the *base* of the arm
    (KUKA mount), not the palm. Accurate palm-center extraction needs
    forward kinematics on the joint positions — for now we approximate
    with the root frame and leave a TODO. This is fine for shape/code
    plumbing but must be replaced before any quantitative claims.
    """
    root = rec["robot_root_states_array"]  # (T, 13)
    pos = root[:, :3]
    quat = root[:, 3:7]
    # TODO: replace with FK on robot_joint_positions_array using the URDF
    # to get the actual palm-center pose.
    aa = quat_xyzw_to_axis_angle(quat)
    return np.concatenate([pos, aa], axis=-1)


def _extract_wrench(rec: dict) -> np.ndarray:
    """Aggregate fingertip + tool-tip forces into a single 6D wrench.

    Order of preference:
    1. ``tool_wrench_array``                       (T, 6)
    2. ``fingertip_wrenches_array`` summed         (T, 5, 6) -> (T, 6)
    3. all zeros, with a warning
    """
    if "tool_wrench_array" in rec:
        return rec["tool_wrench_array"].astype(np.float32)
    if "fingertip_wrenches_array" in rec:
        ft = rec["fingertip_wrenches_array"]  # (T, 5, 6)
        return ft.sum(axis=1).astype(np.float32)
    T = rec["robot_root_states_array"].shape[0]
    warnings.warn(
        "no wrench fields in recording — re-collect with "
        "with_fingertip_force_sensors=True. Filling zeros.",
        stacklevel=2,
    )
    return np.zeros((T, 6), dtype=np.float32)


def _estimate_impedance(
    pose: np.ndarray,
    wrench: np.ndarray,
    dt: float,
    window: int = 20,
    stiffness_floor: float = 1.0,
    damping_floor: float = 0.1,
) -> np.ndarray:
    """Per-step (12,) impedance estimate from regressing wrench against
    pose error and task velocity over a sliding window.

    For each axis i and window of size ``window``:
        F_i ≈ K_i (x_des_i - x_i)  -  D_i x_dot_i

    With x_des set to the window-mean pose, x_dot computed by finite
    differences, we solve a 2-parameter ridge for (K_i, D_i). We then
    floor both to keep the controller well-posed.
    """
    T = pose.shape[0]
    out = np.zeros((T, 12), dtype=np.float32)
    if T < window + 2:
        out[:, :6] = stiffness_floor
        out[:, 6:] = damping_floor
        return out

    vel = np.zeros_like(pose)
    vel[1:] = (pose[1:] - pose[:-1]) / dt

    for t in range(T):
        lo = max(0, t - window)
        hi = min(T, t + 1)
        x = pose[lo:hi]
        v = vel[lo:hi]
        F = wrench[lo:hi]  # (W, 6)
        x_des = x.mean(axis=0, keepdims=True)
        err = x_des - x  # (W, 6)
        for i in range(6):
            A = np.stack([err[:, i], -v[:, i]], axis=-1)  # (W, 2)
            b = F[:, i]
            ATA = A.T @ A + 1e-3 * np.eye(2)
            ATb = A.T @ b
            k_d = np.linalg.solve(ATA, ATb)
            out[t, i] = max(stiffness_floor, float(k_d[0]))
            out[t, 6 + i] = max(damping_floor, float(k_d[1]))
    return out


def extract_labels(npz_path: Path, control_hz: float = 60.0) -> OSLabels:
    """Load a ``.npz`` recording and produce all labels."""
    with np.load(npz_path, allow_pickle=True) as data:
        rec = {k: data[k] for k in data.files}

    pose = _extract_pose(rec)
    wrench = _extract_wrench(rec)
    impedance = _estimate_impedance(pose, wrench, dt=1.0 / control_hz)
    joint_pos = rec["robot_joint_positions_array"].astype(np.float32)
    timestamps = rec["time_array"].astype(np.float64)
    return OSLabels(
        pose=pose.astype(np.float32),
        wrench=wrench.astype(np.float32),
        impedance=impedance.astype(np.float32),
        joint_pos=joint_pos,
        timestamps=timestamps,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    labels = extract_labels(args.npz)
    out = args.out or args.npz.with_suffix(".labels.npz")
    np.savez_compressed(
        out,
        pose=labels.pose,
        wrench=labels.wrench,
        impedance=labels.impedance,
        joint_pos=labels.joint_pos,
        timestamps=labels.timestamps,
    )
    print(f"wrote {out}  T={labels.pose.shape[0]}")


if __name__ == "__main__":
    main()
