import argparse, os
import numpy as np
import torch
import torch.nn.functional as F
from opspace_vla.env import make_env, build_state, PHASE_APPROACH, PHASE_WIPE
from opspace_vla.scripted_policy import scripted_osc_action
from opspace_vla.rollout import load
from opspace_vla.viz import save_png, label_frame, write_video

def rollout_frames(ckpt_dir, kind, camera, img_size, horizon, seed):
    model, cfg, mean, std = load(ckpt_dir)
    cf = int(cfg.get("control_freq", 20))
    vision = bool(cfg.get("vision"))
    env = make_env("osc" if kind == "opspace" else "joint", with_images=True,
                   horizon=horizon, camera=camera, img_size=img_size, control_freq=cf, seed=seed)
    key = f"{camera}_image"
    obs = env.reset(); obs["_phase"] = PHASE_APPROACH
    frames = []
    for _ in range(horizon):
        obs["_phase"] = PHASE_WIPE if bool(obs["robot0_contact"]) else PHASE_APPROACH
        s = (build_state(obs, env) - mean) / std
        batch = {"state": torch.from_numpy(s.astype(np.float32))[None]}
        if vision:
            img = torch.from_numpy(obs[key].copy()).permute(2, 0, 1).float()[None] / 255.0
            if img.shape[-1] != 128:
                img = F.interpolate(img, size=(128, 128), mode="bilinear", align_corners=False)
            batch["image"] = img
        with torch.no_grad():
            pred = model(batch).numpy()[0]
        action = np.clip(pred[:6] if kind == "opspace" else pred[:7], -1, 1)
        obs, _, done, _ = env.step(action)
        frames.append(obs[key].copy())
        if done:
            break
    return frames, float(obs["proportion_wiped"]), cf

def run_compare(args):
    def ok(d):
        return os.path.isfile(os.path.join(d, "config.json")) and os.path.isfile(os.path.join(d, "model.pt"))
    if not (ok(args.opspace_ckpt) and ok(args.joint_ckpt)):
        print(f"need trained checkpoints at '{args.opspace_ckpt}' and '{args.joint_ckpt}'")
        return
    of, oc, ocf = rollout_frames(args.opspace_ckpt, "opspace", args.camera, args.img_size, args.horizon, args.seed)
    jf, jc, jcf = rollout_frames(args.joint_ckpt, "joint", args.camera, args.img_size, args.horizon, args.seed)
    fps = args.fps or ocf
    n = max(len(of), len(jf))
    pick = lambda fr, i: fr[min(i, len(fr) - 1)]
    combined = [np.hstack([label_frame(np.flipud(pick(of, i)), f"op-space  cov={oc:.2f}"),
                           label_frame(np.flipud(pick(jf, i)), f"joint  cov={jc:.2f}")]).astype(np.uint8)
                for i in range(n)]
    path = write_video(combined, os.path.join(args.out, "compare"), fps)
    print(f"op-space {oc:.3f} | joint {jc:.3f}  -> {path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--camera", default="agentview")
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--n-frames", type=int, default=4)
    p.add_argument("--video", action="store_true")
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--compare", action="store_true")
    p.add_argument("--opspace-ckpt", default="checkpoints/opspace")
    p.add_argument("--joint-ckpt", default="checkpoints/joint")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.compare:
        run_compare(args)
        return

    env = make_env("osc", with_images=True, horizon=args.horizon, camera=args.camera, img_size=args.img_size)
    key = f"{args.camera}_image"
    for ep in range(args.episodes):
        np.random.seed(100 + ep)
        obs = env.reset(); obs["_phase"] = 0
        frames, got_contact = [], False
        for t in range(args.horizon):
            a, ph, _ = scripted_osc_action(obs, env)
            obs, _, done, _ = env.step(a); obs["_phase"] = ph
            frames.append(obs[key].copy())
            if not got_contact and bool(obs["robot0_contact"]):
                save_png(obs[key], os.path.join(args.out, f"ep{ep}_first_contact.png"))
                got_contact = True
            if done:
                break
        idx = np.linspace(0, len(frames) - 1, args.n_frames).astype(int)
        for j, fi in enumerate(idx):
            save_png(frames[fi], os.path.join(args.out, f"ep{ep}_frame{j}.png"))
        print(f"ep{ep}: coverage {float(obs['proportion_wiped']):.3f}")
        if args.video:
            fps = args.fps or int(getattr(env, "control_freq", 20))
            name = "rollout" if args.episodes == 1 else f"rollout_ep{ep}"
            vid = [np.flipud(f).astype(np.uint8) for f in frames]
            print("wrote", write_video(vid, os.path.join(args.out, name), fps))

    print(f"done -> {args.out}/")

if __name__ == "__main__":
    main()
