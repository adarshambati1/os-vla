#!/bin/bash
exec > /workspace/run_eval.log 2>&1
set -x
PY=/workspace/miniconda3/envs/libero/bin/python
CT=/workspace/openvla-oft/prismatic/vla/constants.py
RUNS=/workspace/vla_runs
cd /workspace/vla_scripts

N=${1:-50}
SPLITS=("0,1,2" "3,4,5" "6,7" "8,9")

run_rung () {
  rung=$1
  ckpt=$2
  for g in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g MUJOCO_GL=egl $PY eval_rungs.py \
      --rung "$rung" --checkpoint "$ckpt" --num_trials "$N" \
      --tasks "${SPLITS[$g]}" --out "/workspace/eval_${rung}_g${g}.json" &
  done
  wait
  $PY - "$rung" <<"PYMERGE"
import json, glob, sys, numpy as np
rung = sys.argv[1]
per = {}
for f in sorted(glob.glob(f"/workspace/eval_{rung}_g*.json")):
    per.update(json.load(open(f))["per_task"])
def avg(key):
    vals = [v[key] for v in per.values() if v.get(key) is not None]
    return float(np.mean(vals)) if vals else None
out = {
    "rung": rung,
    "overall_success_rate": avg("success_rate"),
    "overall_mean_steps_to_success": avg("mean_steps_to_success"),
    "overall_mean_peak_force": avg("mean_peak_force"),
    "overall_mean_force": avg("mean_force"),
    "overall_force_failure_rate": avg("force_failure_rate"),
    "overall_mean_commanded_fz": avg("mean_commanded_fz"),
    "overall_mean_force_tracking_err": avg("mean_force_tracking_err"),
    "per_task": per,
}
json.dump(out, open(f"/workspace/eval_{rung}.json", "w"), indent=2)
print(f"{rung} overall success: {out['overall_success_rate']:.3f}")
PYMERGE
}

OSC=$(ls -d $RUNS/*libero_osc*--6000_chkpt)
FORCECMD=$(ls -d $RUNS/*libero_forcecmd*--6000_chkpt)
FORCEWRENCH=$(ls -d $RUNS/*libero_forcewrench*--6000_chkpt)
JOINT=$(ls -d $RUNS/*libero_joint*--6000_chkpt)

run_rung osc "$OSC"

sed -i '0,/"PROPRIO_DIM": 8/s//"PROPRIO_DIM": 14/' $CT
sed -i '0,/"ACTION_DIM": 7/s//"ACTION_DIM": 8/' $CT
run_rung forcecmd "$FORCECMD"
sed -i '0,/"PROPRIO_DIM": 14/s//"PROPRIO_DIM": 8/' $CT
sed -i '0,/"ACTION_DIM": 8/s//"ACTION_DIM": 7/' $CT

sed -i '0,/"PROPRIO_DIM": 8/s//"PROPRIO_DIM": 14/' $CT
sed -i '0,/"ACTION_DIM": 7/s//"ACTION_DIM": 13/' $CT
run_rung forcewrench "$FORCEWRENCH"
sed -i '0,/"PROPRIO_DIM": 14/s//"PROPRIO_DIM": 8/' $CT
sed -i '0,/"ACTION_DIM": 13/s//"ACTION_DIM": 7/' $CT

sed -i '0,/"ACTION_DIM": 7/s//"ACTION_DIM": 8/' $CT
run_rung joint "$JOINT"
sed -i '0,/"ACTION_DIM": 8/s//"ACTION_DIM": 7/' $CT

touch /workspace/EVAL_DONE
echo "=== EVAL DONE ==="
