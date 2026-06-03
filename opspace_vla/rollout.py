import json, os
import numpy as np
import torch
from opspace_vla.models import MLPPolicy, VisionPolicy
from opspace_vla.env import make_env, build_state, success, install_hybrid_controller, PHASE_APPROACH, PHASE_WIPE
from opspace_vla.scripted_policy import scripted_osc_action

def load(ckpt_dir):
    cfg = json.load(open(os.path.join(ckpt_dir, "config.json")))
    norm = np.load(os.path.join(ckpt_dir, "norm.npz"))
    if cfg.get("vision"):
        model = VisionPolicy(cfg["out_dim"], hidden=cfg["hidden"])
    else:
        model = MLPPolicy(cfg["in_dim"], cfg["out_dim"], hidden=cfg["hidden"])
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "model.pt"), map_location="cpu"))
    model.eval()
    return model, cfg, norm["state_mean"], norm["state_std"]

def rollout(model, kind, mean, std, episodes, horizon, seed=2000, control_freq=20,
            vision=False, camera="agentview", img_size=128, hybrid=False):
    env = make_env("osc" if kind == "opspace" else "joint", horizon=horizon,
                   control_freq=control_freq, seed=seed,
                   with_images=vision, camera=camera, img_size=img_size)
    key = f"{camera}_image"
    covs, succ = [], []
    for ep in range(episodes):
        obs = env.reset(); obs["_phase"] = PHASE_APPROACH
        if hybrid and kind == "opspace":
            install_hybrid_controller(env) #khattib controller
        for _ in range(horizon):
            obs["_phase"] = PHASE_WIPE if bool(obs["robot0_contact"]) else PHASE_APPROACH
            s = (build_state(obs, env) - mean) / std
            batch = {"state": torch.from_numpy(s.astype(np.float32))[None]}
            if vision:
                batch["image"] = torch.from_numpy(obs[key].copy()).permute(2, 0, 1).float()[None] / 255.0
            with torch.no_grad():
                pred = model(batch).numpy()[0]
            action = np.clip(pred[:6] if kind == "opspace" else pred[:7], -1, 1)
            obs, _, done, _ = env.step(action)
            if done:
                break
        covs.append(float(obs["proportion_wiped"])); succ.append(success(env))
    return np.mean(covs), np.std(covs), np.mean(succ)

def expert_rollout(episodes, horizon, seed=2000, control_freq=20):
    env = make_env("osc", horizon=horizon, control_freq=control_freq, seed=seed)
    covs, succ = [], []
    for _ in range(episodes):
        obs = env.reset(); obs["_phase"] = PHASE_APPROACH
        install_hybrid_controller(env)
        for _ in range(horizon):
            action, phase, _ = scripted_osc_action(obs, env)
            obs, _, done, _ = env.step(action); obs["_phase"] = phase
            if done:
                break
        covs.append(float(obs["proportion_wiped"])); succ.append(success(env))
    return np.mean(covs), np.std(covs), np.mean(succ)
