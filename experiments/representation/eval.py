import argparse, json, os
from opspace_vla.rollout import load, rollout, expert_rollout

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--opspace", required=True)
    p.add_argument("--joint", required=True)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--seed", type=int, default=2000)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--hybrid", action="store_true")
    p.add_argument("--json", default=None)
    args = p.parse_args()

    mo, co, mn_o, sd_o = load(args.opspace)
    mj, cj, mn_j, sd_j = load(args.joint)
    cf = co.get("control_freq", 20)
    vis = bool(co.get("vision"))

    cov_o, std_o, suc_o = rollout(mo, "opspace", mn_o, sd_o, args.episodes, args.horizon,
                                  seed=args.seed, control_freq=cf, vision=vis, img_size=args.img_size, hybrid=args.hybrid)
    print(f"OP-SPACE  coverage {cov_o:.3f} +/- {std_o:.3f}   success {suc_o:.2%}")
    cov_j, std_j, suc_j = rollout(mj, "joint", mn_j, sd_j, args.episodes, args.horizon,
                                  seed=args.seed, control_freq=cf, vision=vis, img_size=args.img_size, hybrid=args.hybrid)
    print(f"JOINT     coverage {cov_j:.3f} +/- {std_j:.3f}   success {suc_j:.2%}")

    cov_e, std_e, suc_e = expert_rollout(args.episodes, args.horizon, seed=args.seed, control_freq=cf)
    print(f"EXPERT    coverage {cov_e:.3f} +/- {std_e:.3f}   success {suc_e:.2%}")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        json.dump({
            "expert":  {"coverage_mean": float(cov_e), "coverage_std": float(std_e), "success_rate": float(suc_e)},
            "opspace": {"coverage_mean": float(cov_o), "coverage_std": float(std_o), "success_rate": float(suc_o)},
            "joint":   {"coverage_mean": float(cov_j), "coverage_std": float(std_j), "success_rate": float(suc_j)},
            "episodes": args.episodes,
        }, open(args.json, "w"), indent=2)
        print("wrote", args.json)

if __name__ == "__main__":
    main()
