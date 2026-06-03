import json, os, glob
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES, FIG = "results", "figures"
os.makedirs(FIG, exist_ok=True)

def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None

def chart_closedloop():
    d = _load(os.path.join(RES, "closedloop.json"))
    if not d:
        return
    labels = ["op-space", "joint"]
    cov = [d["opspace"]["coverage_mean"], d["joint"]["coverage_mean"]]
    err = [d["opspace"]["coverage_std"], d["joint"]["coverage_std"]]
    suc = [d["opspace"]["success_rate"], d["joint"]["success_rate"]]
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    ax[0].bar(labels, cov, yerr=err, capsize=6, color=["#2a7", "#a44"])
    ax[0].set_ylabel("coverage"); ax[0].set_title("coverage")
    ax[1].bar(labels, suc, color=["#2a7", "#a44"])
    ax[1].set_ylabel("success"); ax[1].set_title("success")
    fig.suptitle("op-space vs joint")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "closedloop.png"), dpi=140)
    print("wrote figures/closedloop.png")

def chart_ablation():
    d = _load(os.path.join(RES, "ablation.json"))
    if not d:
        return
    labels = list(d.keys())
    cov = [d[k]["coverage_mean"] for k in labels]
    err = [d[k].get("coverage_std", 0) for k in labels]
    plt.figure(figsize=(5, 3.2))
    plt.bar(labels, cov, yerr=err, capsize=6, color="#37a")
    plt.ylabel("coverage"); plt.title("ablation")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "ablation.png"), dpi=140)
    print("wrote figures/ablation.png")

def chart_force():
    d = _load(os.path.join(RES, "force_tracking.json"))
    if not d:
        return
    tgt = d.get("mean_target", d.get("target_force"))
    meas = d.get("mean_measured_fz", d.get("measured_fz_mean"))
    if tgt is None or meas is None:
        return
    plt.figure(figsize=(4.5, 3.2))
    plt.bar(["target", "measured"], [tgt, meas], color=["#888", "#37a"])
    plt.title("force tracking")
    plt.ylabel("force (N)"); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "force_tracking.png"), dpi=140)
    print("wrote figures/force_tracking.png")

def chart_stiffness():
    d = _load(os.path.join(RES, "stiffness_curves.json"))
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
    plt.savefig(os.path.join(FIG, "stiffness_curves.png"), dpi=140)
    print("wrote figures/stiffness_curves.png")

def chart_training():
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
    plt.savefig(os.path.join(FIG, "training.png"), dpi=140)
    print("wrote figures/training.png")

if __name__ == "__main__":
    chart_closedloop(); chart_ablation(); chart_force(); chart_stiffness(); chart_training()
    print("done -> figures/")
