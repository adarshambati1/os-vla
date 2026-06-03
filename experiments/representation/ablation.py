import os, sys, json, argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from opspace_vla.rollout import load, rollout

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True,
                   help="op-space checkpoint dirs, e.g. checkpoints/op6 checkpoints/op12 checkpoints/op18")
    p.add_argument("--labels", nargs="+", default=None,
                   help="labels matching --ckpts order (default: directory names)")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--seed", type=int, default=2000,
                   help="env seed shared by all variants so episodes are paired")
    p.add_argument("--out", default="results/ablation.json")
    args = p.parse_args()

    labels = args.labels or [os.path.basename(c.rstrip("/\\")) for c in args.ckpts]
    if len(labels) != len(args.ckpts):
        raise SystemExit("--labels must match --ckpts in number")

    res = {}
    for ckpt, label in zip(args.ckpts, labels):
        model, cfg, mean, std = load(ckpt)
        cf = cfg.get("control_freq", 20)
        cov, std_, suc = rollout(model, "opspace", mean, std,
                                 args.episodes, args.horizon, seed=args.seed, control_freq=cf)
        res[label] = {"coverage_mean": float(cov), "coverage_std": float(std_),
                      "success_rate": float(suc)}
        print(f"  {label}: coverage {cov:.3f} +/- {std_:.3f}  success {suc:.1%}  (out_dim={cfg['out_dim']})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.out)

if __name__ == "__main__":
    main()
