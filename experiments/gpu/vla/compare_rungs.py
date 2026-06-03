import json
import glob

RUNGS = ["osc", "forcecmd", "forcewrench", "joint"]


def load(rung):
    d = {}
    for f in glob.glob(f"/workspace/eval_{rung}_g*.json"):
        for k, v in json.load(open(f))["per_task"].items():
            d[k] = v
    return d


data = {r: load(r) for r in RUNGS}
osc, fw = data["osc"], data["forcewrench"]
tasks = sorted(osc)


def fmt(v, nd=2):
    return f"{v:.{nd}f}" if v is not None else " -- "


print("PER-TASK  osc vs forcewrench   (succ / steps-to-success / meanForce N / failRate)")
print("-" * 92)
succ_wins, speed_wins, force_wins = [], [], []
for k in tasks:
    o = osc[k]
    f = fw.get(k)
    name = k.replace("KITCHEN_SCENE", "S")[:34]
    if f is None:
        print(f"{name:34s}  osc succ={o['success_rate']:.2f}   | forcewrench: PENDING")
        continue
    os_, fs = o["success_rate"], f["success_rate"]
    ost, fst = o["mean_steps_to_success"], f["mean_steps_to_success"]
    osf, fsf = o["mean_force"], f["mean_force"]
    ofr, ffr = o["force_failure_rate"], f["force_failure_rate"]
    tags = []
    if fs > os_ + 1e-9:
        tags.append("SUCC+"); succ_wins.append(k)
    if ost is not None and fst is not None and fst < ost - 1e-9:
        tags.append("FASTER"); speed_wins.append(k)
    if ffr < ofr - 1e-9:
        tags.append("LESS-FORCE-FAIL"); force_wins.append(k)
    print(f"{name:34s}  succ {os_:.2f}/{fs:.2f}  steps {fmt(ost,0)}/{fmt(fst,0)}  "
          f"meanF {fmt(osf,1)}/{fmt(fsf,1)}  failR {ofr:.2f}/{ffr:.2f}  {' '.join(tags)}")

print("-" * 92)
print(f"forcewrench BEATS osc on SUCCESS in {len(succ_wins)} tasks: {[t[:24] for t in succ_wins]}")
print(f"forcewrench FASTER than osc in {len(speed_wins)} tasks: {[t[:24] for t in speed_wins]}")
print(f"forcewrench fewer force-failures in {len(force_wins)} tasks: {[t[:24] for t in force_wins]}")
for r in RUNGS:
    d = data[r]
    if d:
        rate = sum(v["success_rate"] for v in d.values()) / len(d)
        print(f"  [{r}] {len(d)}/10 tasks done, overall-so-far success={rate:.3f}")

both = [k for k in tasks if k in fw]
if both:
    oracle = sum(max(osc[k]["success_rate"], fw[k]["success_rate"]) for k in both) / len(both)
    def route(k):
        return fw if ("open" in k.lower() and "drawer" in k.lower()) else osc
    rule = sum(route(k)[k]["success_rate"] for k in both) / len(both)
    osc_only = sum(osc[k]["success_rate"] for k in both) / len(both)
    print(f"\n  ROUTER  osc-only={osc_only:.3f}  oracle(max per task)={oracle:.3f}  "
          f"instruction-rule(open-drawer->fw)={rule:.3f}")
