import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from experiments.stiffness.robomimic import TASKS, build_state

def _img_env(task, seed, horizon, kp=150.0):
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config
    cfg = load_composite_controller_config(controller="BASIC", robot="Panda")
    cfg["body_parts"]["right"]["type"] = "OSC_POSE"; cfg["body_parts"]["right"]["impedance_mode"] = "fixed"
    cfg["body_parts"]["right"]["kp"] = kp
    return suite.make(TASKS[task]["env"], robots="Panda", controller_configs=cfg, has_renderer=False,
                      has_offscreen_renderer=True, use_camera_obs=True, camera_names="frontview",
                      camera_heights=320, camera_widths=320, use_object_obs=True,
                      horizon=horizon, control_freq=20, seed=seed)

def main():
    import h5py, torch, imageio
    from PIL import Image, ImageDraw
    from opspace_vla.models import MLPPolicy
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=list(TASKS))
    p.add_argument("--max-demos", type=int, default=40)
    p.add_argument("--horizon", type=int, default=300)
    p.add_argument("--out", default="figures")
    args = p.parse_args()

    ck = f"checkpoints/rmbc_{args.task}"
    cfg = json.load(open(os.path.join(ck, "config.json")))
    norm = np.load(os.path.join(ck, "norm.npz")); mean, std = norm["state_mean"], norm["state_std"]
    m = MLPPolicy(cfg["in_dim"], cfg["out_dim"], hidden=cfg["hidden"])
    m.load_state_dict(torch.load(os.path.join(ck, "model.pt"), map_location="cpu")); m.eval()
    f = h5py.File(TASKS[args.task]["data"], "r")

    def frame(env):
        return np.flipud(env.sim.render(camera_name="frontview", width=320, height=320)).astype(np.uint8)

    def label(fr, txt):
        im = Image.fromarray(fr); ImageDraw.Draw(im).text((6, 6), txt, fill=(255, 255, 0)); return np.asarray(im)

    def human(states, acts):
        env = _img_env(args.task, 0, len(acts) + 5); env.reset()
        env.sim.set_state_from_flattened(states[0]); env.sim.forward()
        fr = [frame(env)]
        for a in acts:
            env.step(a); fr.append(frame(env))
        return fr, bool(env._check_success())

    def policy(states):
        env = _img_env(args.task, 0, args.horizon); env.reset()
        env.sim.set_state_from_flattened(states[0]); env.sim.forward()
        obs = env._get_observations(); fr = [frame(env)]
        for _ in range(args.horizon):
            s = (build_state(obs, args.task) - mean) / std
            with torch.no_grad():
                a = m({"state": torch.from_numpy(s[None].astype(np.float32))}).numpy()[0]
            obs, _, done, _ = env.step(np.clip(a, -1, 1)); fr.append(frame(env))
            if done:
                break
        return fr, bool(env._check_success())

    for di in range(args.max_demos):
        d = f["data"][f"demo_{di}"]; states = d["states"][:]; acts = d["actions"][:]
        hf, hs = human(states, acts)
        if not hs:
            continue
        pf, ps = policy(states)
        if ps:
            n = max(len(hf), len(pf)); pk = lambda fr, i: fr[min(i, len(fr) - 1)]
            comb = [np.hstack([label(pk(hf, i), f"Human demo ({args.task})"),
                               label(pk(pf, i), "Learned BC policy")]) for i in range(n)]
            path = os.path.join(args.out, f"{args.task}_human_vs_policy.mp4")
            imageio.mimsave(path, comb, fps=20, macro_block_size=1)
            print(f"[{args.task}] demo {di}: both succeed -> {path} ({n} frames)")
            break
        print(f"[{args.task}] demo {di}: human ok, policy missed")
    f.close()

if __name__ == "__main__":
    main()
