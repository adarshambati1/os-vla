import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

TASKS = {
    "lift":   dict(env="Lift",             data="data/lift_ph_low_dim.hdf5",
                   objkeys=["cube_pos", "cube_quat", "gripper_to_cube_pos"]),
    "can":    dict(env="PickPlaceCan",     data="data/can_ph_low_dim.hdf5",
                   objkeys=["Can_pos", "Can_quat", "Can_to_robot0_eef_pos", "Can_to_robot0_eef_quat"]),
    "square": dict(env="NutAssemblySquare", data="data/square_ph_low_dim.hdf5",
                   objkeys=["SquareNut_pos", "SquareNut_quat", "SquareNut_to_robot0_eef_pos", "SquareNut_to_robot0_eef_quat"]),
    "tool_hang": dict(env="ToolHang", data="data/tool_hang_ph_low_dim.hdf5",
                   objkeys=["base_pos", "base_quat", "base_to_robot0_eef_pos", "base_to_robot0_eef_quat",
                            "frame_pos", "frame_quat", "frame_to_robot0_eef_pos", "frame_to_robot0_eef_quat",
                            "tool_pos", "tool_quat", "tool_to_robot0_eef_pos", "tool_to_robot0_eef_quat",
                            "frame_is_assembled", "tool_on_frame"]),
}

def obj_from_env(obs, task):
    if task == "lift":
        rel = np.asarray(obs["robot0_eef_pos"], np.float32) - np.asarray(obs["cube_pos"], np.float32)
        return np.concatenate([np.asarray(obs["cube_pos"], np.float32).ravel(),
                               np.asarray(obs["cube_quat"], np.float32).ravel(), rel])
    return np.concatenate([np.asarray(obs[k], np.float32).ravel() for k in TASKS[task]["objkeys"]])

def build_state(obs, task):
    obj = obs["object"] if "object" in obs else obj_from_env(obs, task)
    return np.concatenate([
        np.asarray(obs["robot0_eef_pos"], np.float32).ravel(),
        np.asarray(obs["robot0_eef_quat"], np.float32).ravel(),
        np.asarray(obs["robot0_gripper_qpos"], np.float32).ravel(),
        np.asarray(obj, np.float32).ravel()]).astype(np.float32)

def load_demos(task):
    import h5py
    f = h5py.File(TASKS[task]["data"], "r")
    demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
    S, A, ep = [], [], []
    for i, dn in enumerate(demos):
        o = f["data"][dn]["obs"]; n = f["data"][dn]["actions"].shape[0]
        s = np.concatenate([o["robot0_eef_pos"][:], o["robot0_eef_quat"][:],
                            o["robot0_gripper_qpos"][:], o["object"][:]], axis=1).astype(np.float32)
        S.append(s); A.append(np.asarray(f["data"][dn]["actions"][:], np.float32)); ep.append(np.full(n, i))
    f.close()
    return np.concatenate(S), np.concatenate(A), np.concatenate(ep)

def make_env(task, seed=None, horizon=400, kp=150.0):
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config
    cfg = load_composite_controller_config(controller="BASIC", robot="Panda")
    cfg["body_parts"]["right"]["type"] = "OSC_POSE"
    cfg["body_parts"]["right"]["impedance_mode"] = "fixed"
    cfg["body_parts"]["right"]["kp"] = kp
    return suite.make(TASKS[task]["env"], robots="Panda", controller_configs=cfg,
                      has_renderer=False, use_camera_obs=False, use_object_obs=True,
                      horizon=horizon, control_freq=20, seed=seed)

def ckdir(task):
    return f"checkpoints/rmbc_{task}"

def check(task):
    S, A, ep = load_demos(task)
    mean, std = S.mean(0), S.std(0) + 1e-6
    env = make_env(task, seed=7, horizon=50); obs = env.reset()
    for _ in range(3):
        obs, _, _, _ = env.step(np.zeros(env.action_dim))
    z = (build_state(obs, task) - mean) / std
    no = 9
    print(f"[{task}] state_dim={S.shape[1]} action_dim={A.shape[1]} demos={len(np.unique(ep))} "
          f"| object dims |z|>3: {int(np.sum(np.abs(z[no:])>3))}/{len(z)-no}  max|z|={np.abs(z[no:]).max():.1f}")

def train(task, epochs=60, device=None):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from opspace_vla.models import MLPPolicy
    from opspace_vla.device import pick_device
    from opspace_vla.learning import fit

    class DictDS(Dataset):
        def __init__(self, X, Y):
            self.X, self.Y = torch.tensor(X), torch.tensor(Y)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, i):
            return {"state": self.X[i], "target": self.Y[i]}

    S, A, ep = load_demos(task)
    rng = np.random.RandomState(0); eps = np.unique(ep); rng.shuffle(eps)
    vset = set(eps[:max(1, int(0.15 * len(eps)))].tolist())
    vm = np.array([e in vset for e in ep]); tm = ~vm
    mean, std = S[tm].mean(0), S[tm].std(0) + 1e-6
    dev = pick_device(device)
    model = MLPPolicy(S.shape[1], A.shape[1], hidden=256, n_layers=3, p_drop=0.05).to(dev)
    trL = DataLoader(DictDS((S[tm]-mean)/std, A[tm]), batch_size=256, shuffle=True)
    vaL = DataLoader(DictDS((S[vm]-mean)/std, A[vm]), batch_size=256, shuffle=False)
    out = ckdir(task); os.makedirs(out, exist_ok=True)
    best, _ = fit(model, trL, vaL, lambda pred, tgt: ((pred - tgt) ** 2).mean(),
                  epochs, 3e-4, os.path.join(out, "model.pt"), dev)
    np.savez(os.path.join(out, "norm.npz"), state_mean=mean, state_std=std)
    json.dump({"in_dim": int(S.shape[1]), "out_dim": int(A.shape[1]), "hidden": 256, "best_val": best},
              open(os.path.join(out, "config.json"), "w"), indent=2)
    print(f"[{task}] done. best val {best:.4f}. saved {out}")

def evaluate(task, kp, episodes, horizon):
    import torch
    from opspace_vla.models import MLPPolicy
    out = ckdir(task); cfg = json.load(open(os.path.join(out, "config.json")))
    norm = np.load(os.path.join(out, "norm.npz")); mean, std = norm["state_mean"], norm["state_std"]
    model = MLPPolicy(cfg["in_dim"], cfg["out_dim"], hidden=cfg["hidden"])
    model.load_state_dict(torch.load(os.path.join(out, "model.pt"), map_location="cpu")); model.eval()
    succ = []
    for ep in range(episodes):
        env = make_env(task, seed=1000+ep, horizon=horizon, kp=kp); obs = env.reset()
        for _ in range(horizon):
            s = (build_state(obs, task) - mean) / std
            with torch.no_grad():
                a = model({"state": torch.from_numpy(s[None])}).numpy()[0]
            obs, _, done, _ = env.step(np.clip(a, -1, 1))
            if done: break
        succ.append(bool(env._check_success()))
    sr = float(np.mean(succ))
    print(f"[{task}] kp={kp:>3.0f}  success {sr:.1%}  over {episodes} ep")
    cf = "results/stiffness_curves.json"
    cur = json.load(open(cf)) if os.path.exists(cf) else {}
    cur.setdefault(task, {})[str(int(kp))] = sr
    os.makedirs("results", exist_ok=True)
    json.dump(cur, open(cf, "w"), indent=2)
    return sr

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["check", "train", "eval"], required=True)
    p.add_argument("--task", required=True, choices=list(TASKS))
    p.add_argument("--kp", type=float, default=150.0)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--epochs", type=int, default=60)
    args = p.parse_args()
    if args.mode == "check": check(args.task)
    elif args.mode == "train": train(args.task, epochs=args.epochs)
    else: evaluate(args.task, args.kp, args.episodes, args.horizon)

if __name__ == "__main__":
    main()
