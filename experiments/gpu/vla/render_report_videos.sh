#!/bin/bash
exec > /workspace/render_vids.log 2>&1
set -x
PY=/workspace/miniconda3/envs/libero/bin/python
RUNS=/workspace/vla_runs
cd /workspace/vla_scripts
mkdir -p /workspace/scratch /workspace/vids

declare -A CKPT
CKPT[osc]=$(ls -d $RUNS/*libero_osc*--6000_chkpt)
CKPT[forcecmd]=$(ls -d $RUNS/*libero_forcecmd*--6000_chkpt)
CKPT[forcewrench]=$(ls -d $RUNS/*libero_forcewrench*--6000_chkpt)
CKPT[joint]=$(ls -d $RUNS/*libero_joint*--6000_chkpt)

rend () {
  rung=$1; task=$2; gpu=$3; comp=$4; extra=$5
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl $PY eval_rungs.py \
    --rung "$rung" --checkpoint "${CKPT[$rung]}" --num_trials 2 --tasks "$task" \
    --save_video --video_dir "/workspace/vids/${comp}/${rung}" \
    --out "/workspace/scratch/${rung}_${task}_${comp}.json" $extra \
    > "/workspace/scratch/render_${rung}_${task}_${comp}.log" 2>&1
}

rend osc 0 0 open_top_drawer & sleep 8
rend forcecmd 0 1 open_top_drawer & sleep 8
rend forcewrench 0 3 open_top_drawer & sleep 8
wait
rend joint 0 0 open_top_drawer & sleep 8
rend osc 6 1 open_microwave & sleep 8
rend forcewrench 6 3 open_microwave & sleep 8
wait
rend forcecmd 6 0 open_microwave & sleep 8
rend joint 6 1 open_microwave & sleep 8
rend joint 0 3 joint_1000step "--max_steps 1000" & sleep 8
wait

touch /workspace/VIDS_DONE
echo "=== VIDS DONE $(date) ==="
