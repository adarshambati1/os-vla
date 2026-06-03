import argparse, os, time
import numpy as np
from opspace_vla.env import make_env, build_state, get_wrench, install_hybrid_controller
from opspace_vla.scripted_policy import scripted_osc_action

def collect(episodes, horizon, with_images, img_size, camera, seed0,
            force_expert=False, target_force=40.0, control_freq=20, hybrid=False):
    env = make_env("osc", with_images=with_images, horizon=horizon,
                   camera=camera, img_size=img_size, control_freq=control_freq)
    buf = {k: [] for k in ["state", "phase", "osc_action", "wrench",
                            "impedance", "joint_pos", "next_joint_pos", "ep_id"]}
    if with_images:
        buf["image"] = []
    coverages = []

    if force_expert:
        import robosuite.utils.transform_utils as T
        from opspace_vla.scripted_policy import stiffness_schedule
        from opspace_vla.env import PHASE_WIPE, PHASE_APPROACH

    for ep in range(episodes):
        np.random.seed(seed0 + ep)
        obs = env.reset(); obs["_phase"] = 0
        hyctrl = None
        if hybrid or force_expert:
            hyctrl = install_hybrid_controller(env)
        if force_expert:
            hyctrl.set_force_target([0, 0, -target_force, 0, 0, 0], axes=(2,))

        for t in range(horizon):
            if not force_expert:
                a, phase, imp = scripted_osc_action(obs, env)
            else:
                a, _, _ = scripted_osc_action(obs, env)
                in_c = bool(obs["robot0_contact"])
                a[2] = 0.0
                R = T.quat2mat(obs["robot0_eef_quat"]); w = get_wrench(env)
                hyctrl.set_measured_wrench(np.concatenate([R @ w[:3], R @ w[3:]]))
                phase = PHASE_WIPE if in_c else PHASE_APPROACH
                obs["_phase"] = phase
                imp = stiffness_schedule(in_c)

            s_cur = build_state(obs, env)
            q_cur = np.asarray(obs["robot0_joint_pos"], np.float32)
            img_cur = obs[f"{camera}_image"].astype(np.uint8) if with_images else None

            obs, _, done, _ = env.step(a)
            q_next = np.asarray(obs["robot0_joint_pos"], np.float32)
            w = get_wrench(env)

            buf["state"].append(s_cur)
            buf["phase"].append(phase)
            buf["osc_action"].append(a)
            buf["wrench"].append(w)
            buf["impedance"].append(imp)
            buf["joint_pos"].append(q_cur)
            buf["next_joint_pos"].append(q_next)
            buf["ep_id"].append(ep)
            if with_images:
                buf["image"].append(img_cur)
            if done:
                break

        coverages.append(float(obs["proportion_wiped"]))
        if (ep + 1) % 20 == 0:
            print(f"  ep {ep+1}/{episodes}  mean_coverage={np.mean(coverages):.3f}")

    out = {k: np.asarray(v) for k, v in buf.items()}
    out["mean_coverage"] = np.float32(np.mean(coverages))
    out["control_freq"] = np.int64(control_freq)
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--with-images", action="store_true")
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--camera", default="agentview")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force-expert", action="store_true",
                   help="collect with the force-regulated expert")
    p.add_argument("--target-force", type=float, default=40.0)
    p.add_argument("--control-freq", type=int, default=20,
                   help="policy Hz; 10 narrows the op-space/joint fairness gap")
    p.add_argument("--hybrid", action="store_true",
                   help="collect under the hybrid controller (force off matches plain OSC)")
    p.add_argument("--out", default="data/wipe_demos.npz")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.time()
    data = collect(args.episodes, args.horizon, args.with_images,
                   args.img_size, args.camera, args.seed,
                   force_expert=args.force_expert, target_force=args.target_force,
                   control_freq=args.control_freq, hybrid=args.hybrid)
    np.savez_compressed(args.out, **data)
    n = len(data["state"])
    print(f"saved {n} transitions to {args.out} in {time.time()-t0:.0f}s "
          f"(mean coverage {float(data['mean_coverage']):.3f})")
