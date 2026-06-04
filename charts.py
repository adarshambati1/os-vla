import json, os, glob
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def paths(exp):
    res = f"experiments/{exp}/results"
    fig = f"experiments/{exp}/figures"
    os.makedirs(fig, exist_ok=True)
    return res, fig


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def chart_closedloop():
    res, fig = paths("cpu/representation")
    d = _load(os.path.join(res, "closedloop.json"))
    if not d:
        return
    labels = ["op-space", "joint"]
    cov = [d["opspace"]["coverage_mean"], d["joint"]["coverage_mean"]]
    err = [d["opspace"]["coverage_std"], d["joint"]["coverage_std"]]
    f, ax = plt.subplots(figsize=(4.2, 3.6))
    bars = ax.bar(labels, cov, yerr=err, capsize=8, color=["#2a7", "#a44"])
    ax.set_ylabel("wiping coverage")
    ax.set_title("op-space vs joint")
    ax.set_ylim(0, max(c + e for c, e in zip(cov, err)) + 0.12)
    for b, c in zip(bars, cov):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f"{c:.2f}", ha="center", va="bottom", fontsize=11)
    f.tight_layout(); f.savefig(os.path.join(fig, "closedloop.png"), dpi=150)
    print("wrote", os.path.join(fig, "closedloop.png"))


def chart_ablation():
    res, fig = paths("cpu/representation")
    d = _load(os.path.join(res, "ablation.json"))
    if not d:
        return
    labels = list(d.keys())
    cov = [d[k]["coverage_mean"] for k in labels]
    err = [d[k].get("coverage_std", 0) for k in labels]
    plt.figure(figsize=(5, 3.2))
    plt.bar(labels, cov, yerr=err, capsize=6, color="#37a")
    plt.ylabel("coverage"); plt.title("ablation")
    plt.tight_layout(); plt.savefig(os.path.join(fig, "ablation.png"), dpi=140)
    print("wrote", os.path.join(fig, "ablation.png"))


def chart_force():
    res, fig = paths("cpu/force")
    d = _load(os.path.join(res, "force_tracking.json"))
    if not d:
        return
    tgt = d.get("mean_target", d.get("target_force"))
    meas = d.get("mean_measured_fz", d.get("measured_fz_mean"))
    if tgt is None or meas is None:
        return
    f, ax = plt.subplots(figsize=(4.2, 3.6))
    bars = ax.bar(["commanded", "measured"], [tgt, meas], color=["#888", "#37a"])
    ax.set_ylabel("normal force (N)")
    ax.set_title("force tracking (scripted)")
    ax.set_ylim(0, max(tgt, meas) * 1.25)
    for b, v in zip(bars, [tgt, meas]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(tgt, meas) * 0.02,
                f"{v:.1f}", ha="center", va="bottom", fontsize=11)
    f.tight_layout()
    f.savefig(os.path.join(fig, "force_tracking.png"), dpi=150)
    print("wrote", os.path.join(fig, "force_tracking.png"))


def chart_stiffness():
    res, fig = paths("cpu/stiffness")
    d = _load(os.path.join(res, "stiffness_curves.json"))
    if not d:
        return
    plt.figure(figsize=(5.5, 3.5))
    for task in d:
        kps = sorted(int(k) for k in d[task])
        success = [d[task][str(k)] * 100 for k in kps]
        plt.plot(kps, success, marker="o", label=task)
    plt.xscale("log")
    plt.xlabel("stiffness kp")
    plt.ylabel("success (%)")
    plt.legend()
    plt.title("success vs stiffness")
    plt.tight_layout()
    plt.savefig(os.path.join(fig, "stiffness_curves.png"), dpi=140)
    print("wrote", os.path.join(fig, "stiffness_curves.png"))


def chart_diffusion():
    res, fig = paths("gpu/diffusion")
    d = _load(os.path.join(res, "diffusion_stiffness_curves.json"))
    if not d:
        return
    plt.figure(figsize=(5.5, 3.5))
    for task in d:
        kps = sorted(int(k) for k in d[task])
        success = [d[task][str(k)] * 100 for k in kps]
        plt.plot(kps, success, marker="o", label=task)
    plt.xscale("log")
    plt.xlabel("stiffness kp")
    plt.ylabel("success (%)")
    plt.legend()
    plt.title("diffusion policy: success vs stiffness")
    plt.tight_layout()
    plt.savefig(os.path.join(fig, "diffusion_stiffness_curves.png"), dpi=140)
    print("wrote", os.path.join(fig, "diffusion_stiffness_curves.png"))


def chart_stiffness_compare():
    stiff_res, _ = paths("cpu/stiffness")
    diff_res, diff_fig = paths("gpu/diffusion")
    mlp = _load(os.path.join(stiff_res, "stiffness_curves.json"))
    diff = _load(os.path.join(diff_res, "diffusion_stiffness_curves.json"))
    if not mlp or not diff:
        return
    colors = {"lift": "#1f77b4", "can": "#ff7f0e", "square": "#2ca02c", "tool_hang": "#9467bd"}
    plt.figure(figsize=(6.5, 4))
    for task in ["lift", "can", "square"]:
        c = colors.get(task, "#555555")
        if task in mlp:
            kps = sorted(int(k) for k in mlp[task])
            ys = [mlp[task][str(k)] * 100 for k in kps]
            plt.plot(kps, ys, marker="o", linestyle="-", color=c, label=f"{task} MLP")
        if task in diff:
            kps = sorted(int(k) for k in diff[task])
            ys = [diff[task][str(k)] * 100 for k in kps]
            plt.plot(kps, ys, marker="^", linestyle="--", markersize=8,
                     markerfacecolor="none", color=c, label=f"{task} diffusion")
    plt.xscale("log")
    plt.xlabel("stiffness kp")
    plt.ylabel("success (%)")
    plt.legend(loc="upper right", framealpha=0.95, fontsize=8)
    plt.title("MLP vs diffusion across stiffness")
    plt.tight_layout()
    plt.savefig(os.path.join(diff_fig, "stiffness_compare.png"), dpi=140)
    print("wrote", os.path.join(diff_fig, "stiffness_compare.png"))


def chart_training():
    _, fig = paths("cpu/representation")
    hists = sorted(glob.glob("checkpoints/*/history.json"))
    if not hists:
        return
    plt.figure(figsize=(5.5, 3.5))
    for h in hists:
        name = os.path.basename(os.path.dirname(h))
        hi = json.load(open(h))
        plt.plot(hi["val"], label=name)
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.yscale("log")
    plt.legend(); plt.title("training"); plt.tight_layout()
    plt.savefig(os.path.join(fig, "training.png"), dpi=140)
    print("wrote", os.path.join(fig, "training.png"))


if __name__ == "__main__":
    chart_closedloop(); chart_ablation(); chart_force(); chart_stiffness(); chart_diffusion(); chart_stiffness_compare(); chart_training()
    print("done")
