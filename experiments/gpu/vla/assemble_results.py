import json
import glob
import csv

RUNGS = ["osc", "forcecmd", "forcewrench", "joint"]
METRICS = [
    "success_rate", "successes", "trials", "mean_steps_to_success",
    "mean_peak_force", "max_peak_force", "mean_force", "force_failure_rate",
    "mean_commanded_fz", "mean_force_tracking_err", "mean_action_mag",
]
MEAN_METRICS = [
    "success_rate", "mean_steps_to_success", "mean_peak_force", "max_peak_force",
    "mean_force", "force_failure_rate", "mean_commanded_fz",
    "mean_force_tracking_err", "mean_action_mag",
]


def load(rung):
    d = {}
    for f in glob.glob(f"/workspace/eval_{rung}_g*.json"):
        for k, v in json.load(open(f))["per_task"].items():
            d[k] = v
    return d


data = {r: load(r) for r in RUNGS}

out = {}
for rung in RUNGS:
    per_task = data[rung]
    overall = {}
    for m in MEAN_METRICS:
        vals = [v[m] for v in per_task.values() if v.get(m) is not None]
        overall[m] = sum(vals) / len(vals) if vals else None
    overall["total_successes"] = sum(v["successes"] for v in per_task.values())
    overall["total_trials"] = sum(v["trials"] for v in per_task.values())
    out[rung] = {"overall": overall, "per_task": per_task}

osc, fw = data["osc"], data["forcewrench"]
tasks = [k for k in osc if k in fw]
out["router"] = {
    "osc_only": sum(osc[k]["success_rate"] for k in tasks) / len(tasks),
    "oracle_max_per_task": sum(max(osc[k]["success_rate"], fw[k]["success_rate"]) for k in tasks) / len(tasks),
    "instruction_rule": sum(
        (fw if ("open" in k.lower() and "drawer" in k.lower()) else osc)[k]["success_rate"]
        for k in tasks
    ) / len(tasks),
}

json.dump(out, open("/workspace/vla_eval_results.json", "w"), indent=2)

with open("/workspace/vla_eval_per_task.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["rung", "task"] + METRICS)
    for rung in RUNGS:
        for task, v in sorted(data[rung].items()):
            w.writerow([rung, task] + [v.get(m) for m in METRICS])

for rung in RUNGS:
    o = out[rung]["overall"]
    print(f"{rung:12s} succ={o['success_rate']:.3f} steps={o['mean_steps_to_success']} "
          f"meanF={o['mean_force']:.1f} failR={o['force_failure_rate']:.2f}")
print("router:", {k: round(v, 3) for k, v in out["router"].items()})
print("wrote vla_eval_results.json + vla_eval_per_task.csv")
