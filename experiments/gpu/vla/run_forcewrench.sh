#!/bin/bash
exec > /workspace/run_forcewrench.log 2>&1
set -x
PY=/workspace/miniconda3/envs/libero/bin/python
TR=/workspace/miniconda3/envs/libero/bin/torchrun
CT=/workspace/openvla-oft/prismatic/vla/constants.py
cd /workspace/openvla-oft

echo "=== build libero_forcewrench RLDS ==="
mkdir -p /workspace/rlds_dataset_builder/libero_forcewrench
cp /workspace/vla_scripts/libero_forcewrench_dataset_builder.py /workspace/rlds_dataset_builder/libero_forcewrench/
touch /workspace/rlds_dataset_builder/libero_forcewrench/__init__.py
( cd /workspace/rlds_dataset_builder/libero_forcewrench && $PY -m tensorflow_datasets.scripts.cli.main build )

echo "=== register libero_forcewrench in OXE ==="
$PY /workspace/vla_scripts/patch_oxe_forcewrench.py

echo "=== finetune (PROPRIO_DIM=14, ACTION_DIM=13) ==="
sed -i '0,/"PROPRIO_DIM": 8/s//"PROPRIO_DIM": 14/' $CT
sed -i '0,/"ACTION_DIM": 7/s//"ACTION_DIM": 13/' $CT
WANDB_MODE=offline MUJOCO_GL=egl $TR --standalone --nnodes 1 --nproc-per-node 4 \
  vla-scripts/finetune.py --vla_path openvla/openvla-7b \
  --data_root_dir /root/tensorflow_datasets --dataset_name libero_forcewrench --run_root_dir /workspace/vla_runs \
  --use_l1_regression True --use_diffusion False --use_film False --num_images_in_input 1 \
  --use_proprio True --batch_size 8 --max_steps 6000 --save_freq 6000 --lora_rank 32 \
  --image_aug True --wandb_entity dummy --wandb_project osvla > /workspace/ft_forcewrench.log 2>&1
sed -i '0,/"PROPRIO_DIM": 14/s//"PROPRIO_DIM": 8/' $CT
sed -i '0,/"ACTION_DIM": 13/s//"ACTION_DIM": 7/' $CT

ls -d /workspace/vla_runs/*libero_forcewrench*_chkpt 2>/dev/null | grep -v -- '--6000_chkpt' | xargs -r rm -rf
touch /workspace/FORCEWRENCH_DONE
echo "=== FORCEWRENCH DONE at $(date) ==="
