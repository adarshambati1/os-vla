import os
import glob
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))
VID = os.path.join(HERE, "videos")
FIGDIR = os.path.join(HERE, "figures")
TMP = "/tmp/qf"
os.makedirs(TMP, exist_ok=True)

METHODS = ["osc", "forcecmd", "forcewrench", "joint"]
TITLE = {"osc": "OSC (position)", "forcecmd": "Force-cmd (Fz)",
         "forcewrench": "Force-wrench (6-DOF)", "joint": "Joint-space"}
TASKS = [("open_top_drawer", "open top drawer"), ("open_microwave", "open microwave")]


def frame(task, method):
    vids = glob.glob(os.path.join(VID, task, method, "*ep0.mp4"))
    if not vids:
        return None
    out = os.path.join(TMP, f"{task}__{method}.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-sseof", "-0.2", "-i", vids[0],
                    "-update", "1", "-frames:v", "1", out], check=False)
    return out if os.path.exists(out) else None


for task, nice in TASKS:
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6))
    for ax, m in zip(axes, METHODS):
        p = frame(task, m)
        if p:
            ax.imshow(mpimg.imread(p))
        ax.set_title(TITLE[m], fontsize=17, fontweight="bold")
        ax.axis("off")
    fig.suptitle(f"final rollout frame: {nice}", fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, f"qual_{task}.png"), dpi=130)
    plt.close()

print("wrote qualitative montages to", FIGDIR)
