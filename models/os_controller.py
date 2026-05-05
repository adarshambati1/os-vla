"""Khatib operational-space controller.

Given the model's predicted (pose, wrench, impedance) command, compute
joint torques that realize it on the robot.

Following Khatib, "A unified approach for motion and force control of
robot manipulators: The operational space formulation" (1987):

    F* = Lambda(q) * x_ddot_des  +  mu(q, qd)  +  p(q)
    F  = F* + F_desired
    tau = J(q)^T * F

with x_ddot_des computed from the impedance law

    x_ddot_des = K * (x_des - x)  -  D * x_dot

Lambda is the operational-space inertia (J M^{-1} J^T)^{-1}; mu and p
are the OS Coriolis and gravity terms. We compute them via the standard
projection from joint-space dynamics:

    Lambda      = (J M^{-1} J^T)^{-1}
    J_bar^T     = Lambda J M^{-1}              # dynamically-consistent inverse
    mu          = J_bar^T (C qd) - Lambda Jdot qd
    p           = J_bar^T g

This module is used at *eval* time only — at training we just predict
the (pose, wrench, impedance) and supervise via MSE on the labels.

Arguments are torch tensors; everything is differentiable so the
controller is also usable inside an end-to-end actor-critic loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class OSCInputs:
    # Robot state
    q: torch.Tensor          # (n,)            joint positions
    qd: torch.Tensor         # (n,)            joint velocities
    M: torch.Tensor          # (n, n)          joint-space mass
    C_qd: torch.Tensor       # (n,)            Coriolis * qd
    g: torch.Tensor          # (n,)            gravity in joint space
    J: torch.Tensor          # (6, n)          task Jacobian (palm/tool frame)
    Jdot_qd: torch.Tensor    # (6,)            time-deriv of J times qd

    # Task-space state
    x: torch.Tensor          # (6,)            current pose (pos 3 + axis-angle 3)
    xd: torch.Tensor         # (6,)            current task velocity

    # Command from the policy
    x_des: torch.Tensor      # (6,)            desired pose
    F_des: torch.Tensor      # (6,)            desired feed-forward wrench
    K: torch.Tensor          # (6,)            stiffness diag
    D: torch.Tensor          # (6,)            damping diag


def _solve_psd(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """A^{-1} b for a (small) PSD matrix, with a tiny jitter for safety."""
    n = A.shape[-1]
    eye = torch.eye(n, dtype=A.dtype, device=A.device)
    return torch.linalg.solve(A + 1e-6 * eye, b)


def osc_torques(inp: OSCInputs) -> torch.Tensor:
    """Return joint torques tau (shape (n,))."""
    M_inv = torch.linalg.inv(inp.M + 1e-6 * torch.eye(
        inp.M.shape[-1], dtype=inp.M.dtype, device=inp.M.device
    ))
    JMinv = inp.J @ M_inv                              # (6, n)
    Lambda_inv = JMinv @ inp.J.transpose(-1, -2)       # (6, 6)
    Lambda = torch.linalg.inv(
        Lambda_inv + 1e-6 * torch.eye(
            6, dtype=Lambda_inv.dtype, device=Lambda_inv.device
        )
    )
    Jbar_T = Lambda @ JMinv                            # (6, n)

    # Impedance law
    pose_err = inp.x_des - inp.x
    x_ddot_des = inp.K * pose_err - inp.D * inp.xd

    mu = Jbar_T @ inp.C_qd - Lambda @ inp.Jdot_qd
    p = Jbar_T @ inp.g

    F_star = Lambda @ x_ddot_des + mu + p
    F = F_star + inp.F_des
    tau = inp.J.transpose(-1, -2) @ F
    return tau


def quat_to_axis_angle(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Convert a unit quaternion (xyzw) to a 3D axis-angle vector."""
    x, y, z, w = quat_xyzw.unbind(-1)
    norm_xyz = torch.sqrt(x * x + y * y + z * z).clamp_min(1e-8)
    angle = 2.0 * torch.atan2(norm_xyz, w.abs())
    sign = torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
    axis = torch.stack([x, y, z], dim=-1) / norm_xyz.unsqueeze(-1)
    return axis * (angle * sign).unsqueeze(-1)


def axis_angle_to_quat(aa: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle to unit quaternion (xyzw)."""
    angle = torch.linalg.norm(aa, dim=-1, keepdim=True).clamp_min(1e-8)
    axis = aa / angle
    half = 0.5 * angle
    s = torch.sin(half)
    c = torch.cos(half)
    return torch.cat([axis * s, c], dim=-1)
