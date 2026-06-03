import argparse, json, os, sys
import numpy as np
import robosuite.utils.transform_utils as T

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from opspace_vla.env import (make_env, install_hybrid_controller, get_wrench,
                                    success, PHASE_APPROACH, PHASE_WIPE)
from opspace_vla.scripted_policy import scripted_osc_action

FZ_PRED_IDX, POSE_DX, POSE_DY = 8, 0, 1
NORMAL_AXIS = 2

def _world_wrench(obs, env):
    R = T.quat2mat(obs["robot0_eef_quat"])
    w = get_wrench(env)
    return np.concatenate([R @ w[:3], R @ w[3:]])

def run(mode, force_on, target_force, episodes, horizon, ckpt=None, seed=400, camera="agentview"):
    policy = None
    if mode == "mlp":
        from experiments.force.policies import load_mlp_policy
        policy = load_mlp_policy(ckpt)
    env = make_env("osc", with_images=(policy.with_images if policy else False),
                   horizon=horizon, camera=camera, control_freq=10, seed=seed)
    covs, succ, ferr, targets, fzs = [], [], [], [], []
    for ep in range(episodes):
        obs = env.reset(); obs["_phase"] = PHASE_APPROACH
        ctrl = install_hybrid_controller(env)
        for _ in range(horizon):
            in_c = bool(obs["robot0_contact"])
            obs["_phase"] = PHASE_WIPE if in_c else PHASE_APPROACH

            if mode == "scripted":
                a, _, _ = scripted_osc_action(obs, env)
                tgt = target_force
            else:
                pred = policy.predict(obs, env)
                a = np.clip(pred[:6], -1, 1).astype(np.float32)
                tgt = float(np.clip(abs(pred[FZ_PRED_IDX]), 5.0, 120.0))

            if force_on:
                a[NORMAL_AXIS] = 0.0
                ctrl.set_force_target([0, 0, -tgt, 0, 0, 0], axes=(NORMAL_AXIS,))
                ctrl.set_measured_wrench(_world_wrench(obs, env))

            obs, _, done, _ = env.step(a)
            Fz_world = float(_world_wrench(obs, env)[NORMAL_AXIS])
            if bool(obs["robot0_contact"]):
                fzs.append(abs(Fz_world)); targets.append(tgt); ferr.append(abs(abs(Fz_world) - tgt))
            if done:
                break
        covs.append(float(obs["proportion_wiped"])); succ.append(bool(success(env)))
        print(f"  ep{ep}: coverage {covs[-1]:.3f}  mean|Fz| {np.mean(fzs[-50:]) if fzs else float('nan'):.1f}")

    res = {"mode": mode, "force": "on" if force_on else "off",
           "coverage_mean": float(np.mean(covs)), "coverage_std": float(np.std(covs)),
           "success_rate": float(np.mean(succ)),
           "mean_force_err": float(np.mean(ferr)) if (force_on and ferr) else None,
           "mean_target": float(np.mean(targets)) if (force_on and targets) else None,
           "mean_measured_fz": float(np.mean(fzs)) if fzs else None}
    tag = f"{mode}/force-{res['force']}"
    print(f"{tag}: coverage {res['coverage_mean']:.3f}+/-{res['coverage_std']:.3f}  "
          f"success {res['success_rate']:.0%}  "
          + (f"|Fz-target| {res['mean_force_err']:.1f}N (target~{res['mean_target']:.0f}, meas~{res['mean_measured_fz']:.0f})"
             if force_on else f"(force off; meas Fz~{res['mean_measured_fz']})"))
    return res

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["scripted", "mlp"], required=True)
    p.add_argument("--force", choices=["on", "off"], default="on")
    p.add_argument("--target-force", type=float, default=40.0)
    p.add_argument("--ckpt", help="op-space checkpoint (mlp mode)")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--horizon", type=int, default=250)
    p.add_argument("--seed", type=int, default=400)
    p.add_argument("--json", default=None)
    args = p.parse_args()
    if args.mode == "mlp":
        assert args.ckpt, "--ckpt required in mlp mode"
    res = run(args.mode, args.force == "on", args.target_force, args.episodes, args.horizon,
              ckpt=args.ckpt, seed=args.seed)
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        json.dump(res, open(args.json, "w"), indent=2)
        print("wrote", args.json)

if __name__ == "__main__":
    main()
