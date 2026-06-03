#!/bin/bash
exec > /workspace/run_eval.log 2>&1
set -x
PY=/workspace/miniconda3/envs/libero/bin/python
RUNS=/workspace/vla_runs
cd /workspace/vla_scripts

N=${1:-50}
GPUS=(0 1 3)
SPLITS=("0,1,2,6" "3,4,7" "5,8,9")

declare -A CKPT
CKPT[osc]=$(ls -d $RUNS/*libero_osc*--6000_chkpt)
CKPT[forcecmd]=$(ls -d $RUNS/*libero_forcecmd*--6000_chkpt)
CKPT[forcewrench]=$(ls -d $RUNS/*libero_forcewrench*--6000_chkpt)
CKPT[joint]=$(ls -d $RUNS/*libero_joint*--6000_chkpt)

run_rung () {
  rung=$1
  ckpt=$2
  for i in 0 1 2; do
    g=${GPUS[$i]}
    sp=${SPLITS[$i]}
    CUDA_VISIBLE_DEVICES=$g MUJOCO_GL=egl $PY eval_rungs.py \
      --rung "$rung" --checkpoint "$ckpt" --num_trials "$N" --tasks "$sp" \
      --out "/workspace/eval_${rung}_g${g}.json" > "/workspace/eval_${rung}_g${g}.log" 2>&1 &
    sleep 20
  done
  wait
  $PY - "$rung" <<"PYM"
import json, glob, sys, numpy as np
rung = sys.argv[1]
per = {}
for f in sorted(glob.glob(f"/workspace/eval_{rung}_g*.json")):
    per.update(json.load(open(f))["per_task"])
def avg(k):
    v = [x[k] for x in per.values() if x.get(k) is not None]
    return float(np.mean(v)) if v else None
out = {
    "rung": rung,
    "overall_success_rate": avg("success_rate"),
    "overall_mean_steps_to_success": avg("mean_steps_to_success"),
    "overall_mean_peak_force": avg("mean_peak_force"),
    "overall_mean_force": avg("mean_force"),
    "overall_force_failure_rate": avg("force_failure_rate"),
    "overall_mean_commanded_fz": avg("mean_commanded_fz"),
    "overall_mean_action_mag": avg("mean_action_mag"),
    "per_task": per,
}
json.dump(out, open(f"/workspace/eval_{rung}.json", "w"), indent=2)
print(f"{rung} overall success: {out['overall_success_rate']}")
PYM
  echo "=== $rung DONE at $(date) ==="
}

RUNGS_TO_RUN=${2:-"osc forcecmd forcewrench joint"}
for rung in $RUNGS_TO_RUN; do
  run_rung "$rung" "${CKPT[$rung]}"
done

touch /workspace/EVAL_DONE
echo "=== ALL EVALS DONE at $(date) ==="
