import json
import re
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(HERE, "results")
res = json.load(open(os.path.join(RESDIR, "vla_eval_results.json")))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

METHODS = ["osc", "forcecmd", "forcewrench", "joint"]
LABEL = {
    "osc": "OSC\n(position)",
    "forcecmd": "Force-cmd\n(scalar Fz)",
    "forcewrench": "Force-wrench\n(6-DOF)",
    "joint": "Joint-space",
}
COLORS = {"osc": "#4C72B0", "forcecmd": "#C44E52", "forcewrench": "#55A868", "joint": "#8172B2"}


def short(task):
    m = re.match(r"KITCHEN_SCENE(\d+)_(.*)", task)
    rest = m.group(2).replace("_", " ").replace(" of the cabinet", "").replace(" the ", " ")
    return f"{rest.strip()} (S{m.group(1)})"


tasks = sorted(res["osc"]["per_task"].keys())
labels = [short(t) for t in tasks]

vals = [res[m]["overall"]["success_rate"] for m in METHODS] + [res["router"]["instruction_rule"]]
names = [LABEL[m] for m in METHODS] + ["Adaptive\nrouter"]
cols = [COLORS[m] for m in METHODS] + ["#CCB974"]
fig, ax = plt.subplots(figsize=(7.5, 4.5))
bars = ax.bar(names, vals, color=cols)
ax.set_ylabel("overall success rate")
ax.set_ylim(0, 1)
ax.set_title("VLA method comparison\n(10 contact-rich LIBERO tasks, 50 trials each)")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig_overall_success.png"), dpi=150)
plt.close()


def heatmap(metric, cmap, title, fname, fmt="{:.2f}"):
    M = np.array([[res[m]["per_task"][t][metric] if res[m]["per_task"][t][metric] is not None else np.nan
                   for t in tasks] for m in METHODS])
    fig, ax = plt.subplots(figsize=(14, 5.2))
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=0, vmax=(1 if metric == "success_rate" else None))
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=15)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([LABEL[m].replace("\n", " ") for m in METHODS], fontsize=16)
    for i in range(len(METHODS)):
        for j in range(len(tasks)):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=15, fontweight="bold")
    ax.set_title(title, fontsize=19)
    cb = fig.colorbar(im, ax=ax, fraction=0.025)
    cb.ax.tick_params(labelsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, fname), dpi=200)
    plt.close()


heatmap("success_rate", "RdYlGn", "Per-task success rate by method", "fig_success_heatmap.png")
heatmap("mean_force", "magma", "Per-task mean contact force (N) by method", "fig_force_heatmap.png", "{:.0f}")

lines = ["| task | " + " | ".join(METHODS) + " |", "|" + "---|" * (len(METHODS) + 1)]
for t, lab in zip(tasks, labels):
    row = [f"{res[m]['per_task'][t]['success_rate']:.2f}" for m in METHODS]
    lines.append(f"| {lab} | " + " | ".join(row) + " |")
overall = [f"{res[m]['overall']['success_rate']:.2f}" for m in METHODS]
lines.append(f"| **overall** | " + " | ".join(f"**{x}**" for x in overall) + " |")
open(os.path.join(RESDIR, "vla_per_task_table.md"), "w").write("\n".join(lines) + "\n")

print("wrote figures to", FIGDIR)
print("wrote results/vla_per_task_table.md")
