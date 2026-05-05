"""CPU smoke test.

Exercises the full forward + loss + backward path for both the OS and
joint-space heads against the dummy backbone. No GPU, no PaliGemma
weights, no Isaac Gym.

Run:
    python -m tests.smoke
"""

from __future__ import annotations

import torch

from models.action_head import IMPEDANCE_DIM, OS_DIM, POSE_DIM, WRENCH_DIM
from models.os_controller import OSCInputs, osc_torques
from models.osvla import OSVLA, OSVLAConfig
from training.losses import LossWeights, joint_loss, os_loss


def _dummy_batch(B: int, vocab: int = 257152, image_size: int = 224):
    return {
        "input_ids": torch.randint(0, vocab, (B, 8)),
        "pixel_values": torch.randn(B, 3, image_size, image_size),
        "attention_mask": torch.ones(B, 8, dtype=torch.long),
        "pose_target": torch.randn(B, POSE_DIM),
        "wrench_target": torch.randn(B, WRENCH_DIM),
        "impedance_target": torch.rand(B, IMPEDANCE_DIM) + 0.5,
        "joint_target": torch.randn(B, 29),
    }


def test_os_head() -> None:
    cfg = OSVLAConfig(output_space="operational")
    model = OSVLA.dummy(cfg)
    batch = _dummy_batch(B=4)

    pred = model(
        input_ids=batch["input_ids"],
        pixel_values=batch["pixel_values"],
        attention_mask=batch["attention_mask"],
    )
    assert pred["pose"].shape == (4, POSE_DIM)
    assert pred["wrench"].shape == (4, WRENCH_DIM)
    assert pred["impedance"].shape == (4, IMPEDANCE_DIM)
    assert (pred["impedance"] > 0).all(), "impedance must be positive"

    loss, parts = os_loss(
        pred,
        {
            "pose": batch["pose_target"],
            "wrench": batch["wrench_target"],
            "impedance": batch["impedance_target"],
        },
        LossWeights(),
    )
    loss.backward()
    assert torch.isfinite(loss)
    print(f"  os: loss={loss.item():.4f}  parts={ {k: round(v.item(), 4) for k, v in parts.items()} }")


def test_joint_head() -> None:
    cfg = OSVLAConfig(output_space="joint", joint_dim=29)
    model = OSVLA.dummy(cfg)
    batch = _dummy_batch(B=4)

    pred = model(
        input_ids=batch["input_ids"],
        pixel_values=batch["pixel_values"],
        attention_mask=batch["attention_mask"],
    )
    assert pred.shape == (4, 29)

    loss, parts = joint_loss(pred, batch["joint_target"])
    loss.backward()
    assert torch.isfinite(loss)
    print(f"  joint: loss={loss.item():.4f}")


def test_os_dim_constants() -> None:
    assert OS_DIM == POSE_DIM + WRENCH_DIM + IMPEDANCE_DIM == 24
    print("  dims: pose=6 wrench=6 impedance=12 total=24")


def test_osc_torques() -> None:
    n = 7
    rng = torch.Generator().manual_seed(0)
    M = torch.eye(n) + 0.1 * torch.randn(n, n, generator=rng)
    M = M @ M.T  # PSD
    inp = OSCInputs(
        q=torch.zeros(n),
        qd=torch.zeros(n),
        M=M,
        C_qd=torch.zeros(n),
        g=torch.zeros(n),
        J=torch.randn(6, n, generator=rng),
        Jdot_qd=torch.zeros(6),
        x=torch.zeros(6),
        xd=torch.zeros(6),
        x_des=torch.tensor([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
        F_des=torch.zeros(6),
        K=torch.full((6,), 200.0),
        D=torch.full((6,), 20.0),
    )
    tau = osc_torques(inp)
    assert tau.shape == (n,)
    assert torch.isfinite(tau).all()
    print(f"  osc: tau={tau.detach().numpy().round(3).tolist()}")


def test_extract_os_labels(tmp_dir: str = "/tmp/osvla_smoke") -> None:
    import os
    import numpy as np

    from data.scripts.extract_os_labels import extract_labels

    os.makedirs(tmp_dir, exist_ok=True)
    T = 30
    rec_path = os.path.join(tmp_dir, "fake.npz")
    np.savez(
        rec_path,
        robot_root_states_array=np.random.randn(T, 13).astype(np.float64),
        object_root_states_array=np.random.randn(T, 13).astype(np.float64),
        robot_joint_positions_array=np.random.randn(T, 29).astype(np.float64),
        time_array=np.arange(T) / 60.0,
        # No wrench fields -> should warn and produce zeros.
    )
    labels = extract_labels(rec_path)
    assert labels.pose.shape == (T, 6)
    assert labels.wrench.shape == (T, 6)
    assert labels.impedance.shape == (T, 12)
    assert labels.joint_pos.shape == (T, 29)
    print(
        f"  labels: pose={labels.pose.shape} wrench={labels.wrench.shape} "
        f"imp={labels.impedance.shape} joint={labels.joint_pos.shape}"
    )


def main() -> None:
    print("smoke tests")
    test_os_dim_constants()
    test_os_head()
    test_joint_head()
    test_osc_torques()
    test_extract_os_labels()
    print("all green.")


if __name__ == "__main__":
    main()
