#!/bin/bash
exec > /workspace/rebuild.log 2>&1
set -x
rm -f /workspace/REBUILD_DONE
echo "=== [1/12] apt GL libs + tmux ==="
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libegl1 libgles2 libglvnd0 libgl1 libosmesa6 tmux wget git >/dev/null 2>&1 || true
echo "=== [2/12] miniconda ==="
[ -d /workspace/miniconda3 ] || (wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh && bash /tmp/mc.sh -b -p /workspace/miniconda3)
echo "=== [3/12] clone repos ==="
cd /workspace
[ -d openvla-oft ] || git clone -q https://github.com/moojink/openvla-oft.git
[ -d LIBERO ] || git clone -q https://github.com/Lifelong-Robot-Learning/LIBERO.git
[ -d rlds_dataset_builder ] || git clone -q https://github.com/kpertsch/rlds_dataset_builder.git
echo "=== [4/12] conda env ==="
PY=/workspace/miniconda3/envs/libero/bin/python
[ -x "$PY" ] || /workspace/miniconda3/bin/conda create -y -q -n libero -c conda-forge --override-channels python=3.10
$PY -m ensurepip --upgrade
echo "=== [5/12] torch cu128 ==="
$PY -m pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
echo "=== [6/12] OFT package ==="
$PY -m pip install -q -e /workspace/openvla-oft --no-deps
echo "=== [7/12] OFT deps ==="
$PY -m pip install -q "git+https://github.com/moojink/transformers-openvla-oft" "git+https://github.com/moojink/dlimp_openvla" tensorflow==2.15.0 tensorflow_datasets==4.9.3 tensorflow_graphics timm==0.9.10 "tokenizers==0.19.1" "peft==0.11.1" accelerate draccus rich json-numpy "sentencepiece==0.1.99" "diffusers==0.30.3" jsonlines imageio imageio-ffmpeg opencv-python matplotlib einops huggingface_hub termcolor numba wandb
echo "=== [8/12] robosuite 1.4 + LIBERO ==="
$PY -m pip install -q "robosuite==1.4.0" "bddl==1.0.1" easydict "numpy<2" h5py future cloudpickle gym hydra-core thop
$PY -m pip install -q -e /workspace/LIBERO --no-deps
echo "=== [9/12] LIBERO config init ==="
printf "N\nN\nN\nN\n" | $PY -c "import libero; from libero.libero import benchmark; benchmark.get_benchmark_dict()" || true
echo "=== [10/12] OXE patches ==="
$PY /workspace/vla_scripts/patch_oxe.py libero_osc
$PY /workspace/vla_scripts/patch_oxe_force.py
echo "=== [11/12] download demos + wrench ==="
$PY - <<'PYEOF'
from huggingface_hub import snapshot_download
tasks=["KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet","KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet","KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet","KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet","KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet","KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet","KITCHEN_SCENE7_open_the_microwave","KITCHEN_SCENE6_close_the_microwave","KITCHEN_SCENE3_turn_on_the_stove","KITCHEN_SCENE8_turn_off_the_stove"]
snapshot_download(repo_id="yifengzhu-hf/LIBERO-datasets", repo_type="dataset", allow_patterns=[f"libero_90/{t}_demo.hdf5" for t in tasks], local_dir="/workspace/LIBERO/libero/datasets")
print("demos downloaded")
PYEOF
cp /workspace/vla_scripts/extract_wrench.py /tmp/
for i in $(seq 0 9); do MUJOCO_GL=egl $PY /tmp/extract_wrench.py $i; done
echo "=== [12/12] build RLDS ==="
mkdir -p /workspace/rlds_dataset_builder/libero_osc /workspace/rlds_dataset_builder/libero_force
cp /workspace/vla_scripts/libero_osc_dataset_builder.py /workspace/rlds_dataset_builder/libero_osc/
cp /workspace/vla_scripts/libero_force_dataset_builder.py /workspace/rlds_dataset_builder/libero_force/
touch /workspace/rlds_dataset_builder/libero_osc/__init__.py /workspace/rlds_dataset_builder/libero_force/__init__.py
cd /workspace/rlds_dataset_builder/libero_osc && $PY -m tensorflow_datasets.scripts.cli.main build
cd /workspace/rlds_dataset_builder/libero_force && $PY -m tensorflow_datasets.scripts.cli.main build
echo "REBUILD_DONE"
touch /workspace/REBUILD_DONE
