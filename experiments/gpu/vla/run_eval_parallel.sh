#!/bin/bash
exec > /workspace/run_eval.log 2>&1
set -x
PY=/workspace/miniconda3/envs/libero/bin/python
RUNS=/workspace/vla_runs
cd /workspace/vla_scripts

N=${1:-50}

declare -A CKPT
CKPT[osc]=$(ls -d $RUNS/*libero_osc*--6000_chkpt)
CKPT[forcecmd]=$(ls -d $RUNS/*libero_forcecmd*--6000_chkpt)
CKPT[forcewrench]=$(ls -d $RUNS/*libero_forcewrench*--6000_chkpt)
CKPT[joint]=$(ls -d $RUNS/*libero_joint*--6000_chkpt)
RUNGS=(osc forcecmd forcewrench joint)

rm -f /workspace/EVAL_DONE
for g in 0 1 2 3; do
  rung=${RUNGS[$g]}
  CUDA_VISIBLE_DEVICES=$g MUJOCO_GL=egl $PY eval_rungs.py \
    --rung "$rung" --checkpoint "${CKPT[$rung]}" --num_trials "$N" \
    --out "/workspace/eval_${rung}.json" > "/workspace/eval_${rung}.log" 2>&1 &
  sleep 20
done
wait

touch /workspace/EVAL_DONE
echo "=== ALL EVALS DONE at $(date) ==="
