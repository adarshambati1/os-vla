import argparse, os, json
import numpy as np
import torch
from torch.utils.data import DataLoader

from opspace_vla.dataset import WipeDataset, opspace_loss_weights, OPSPACE_DIM
from opspace_vla.models import MLPPolicy, VisionPolicy
from opspace_vla.device import pick_device
from opspace_vla.learning import fit

def make_loss(target, dims, device, opspace_targets=None):
    if target == "opspace":
        base = opspace_loss_weights().clone()
        if opspace_targets is not None:
            t = np.asarray(opspace_targets, np.float32)
            for sl in (slice(0, 6), slice(6, 12), slice(12, 18)):
                base[sl] = base[sl] / (float(np.mean(t[:, sl] ** 2)) + 1e-8)
        w = base[:dims].to(device)
        return lambda pred, tgt: (w * (pred - tgt[:, :dims]) ** 2).mean()
    return lambda pred, tgt: ((pred - tgt) ** 2).mean()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--target", choices=["opspace", "joint"], required=True)
    p.add_argument("--opspace-dims", type=int, default=OPSPACE_DIM)
    p.add_argument("--vision", action="store_true")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)

    tr = WipeDataset(args.data, args.target, split="train", has_images=args.vision)
    va = WipeDataset(args.data, args.target, split="val", has_images=args.vision,
                     state_mean=tr.state_mean, state_std=tr.state_std)
    out_dim = args.opspace_dims if args.target == "opspace" else 7
    print(f"[{args.target}] train={len(tr)} val={len(va)} out_dim={out_dim} device={device}")

    if args.vision:
        model = VisionPolicy(out_dim, hidden=args.hidden, p_drop=args.dropout).to(device)
    else:
        model = MLPPolicy(tr.state.shape[1], out_dim, hidden=args.hidden, p_drop=args.dropout).to(device)
    loss_fn = make_loss(args.target, out_dim, device,
                        opspace_targets=(tr.opspace if args.target == "opspace" else None))

    trL = DataLoader(tr, batch_size=args.batch, shuffle=True)
    vaL = DataLoader(va, batch_size=args.batch, shuffle=False)
    best, hist = fit(model, trL, vaL, loss_fn, args.epochs, args.lr,
                     os.path.join(args.out, "model.pt"), device)

    np.savez(os.path.join(args.out, "norm.npz"), state_mean=tr.state_mean, state_std=tr.state_std)
    d = np.load(args.data)
    control_freq = int(d["control_freq"]) if "control_freq" in d.files else 20
    json.dump({**vars(args), "in_dim": int(tr.state.shape[1]), "out_dim": out_dim,
               "control_freq": control_freq, "best_val": best},
              open(os.path.join(args.out, "config.json"), "w"), indent=2)
    json.dump(hist, open(os.path.join(args.out, "history.json"), "w"))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.plot(hist["train"], label="train"); plt.plot(hist["val"], label="val")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title(args.target)
        plt.savefig(os.path.join(args.out, "loss.png"), dpi=120, bbox_inches="tight")
    except Exception as e:
        print("plot skipped:", e)

    print(f"done. best val {best:.4f}. saved to {args.out}")

if __name__ == "__main__":
    main()
