#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
py="conda run -n opspace-vla python"

mkdir -p experiments/cpu/representation/results experiments/cpu/force/results experiments/cpu/stiffness/results

$py experiments/cpu/representation/collect_data.py --episodes 200 --horizon 250 --control-freq 10 --hybrid --out data/wipe_demos.npz
$py experiments/cpu/representation/train.py --data data/wipe_demos.npz --target opspace --out checkpoints/opspace
$py experiments/cpu/representation/train.py --data data/wipe_demos.npz --target joint --out checkpoints/joint
$py experiments/cpu/representation/eval.py --opspace checkpoints/opspace --joint checkpoints/joint --episodes 30 --hybrid --json experiments/cpu/representation/results/closedloop.json

$py experiments/cpu/representation/collect_data.py --episodes 60 --horizon 250 --control-freq 10 --with-images --img-size 128 --hybrid --out data/wipe_img.npz
$py experiments/cpu/representation/train.py --data data/wipe_img.npz --target opspace --vision --device mps --out checkpoints/opspace_vis
$py experiments/cpu/representation/train.py --data data/wipe_img.npz --target joint --vision --device mps --out checkpoints/joint_vis
$py experiments/cpu/representation/eval.py --opspace checkpoints/opspace_vis --joint checkpoints/joint_vis --episodes 30 --img-size 128 --hybrid --json experiments/cpu/representation/results/closedloop_vision.json

$py experiments/cpu/representation/train.py --data data/wipe_demos.npz --target opspace --opspace-dims 6 --out checkpoints/op6
$py experiments/cpu/representation/train.py --data data/wipe_demos.npz --target opspace --opspace-dims 12 --out checkpoints/op12
$py experiments/cpu/representation/train.py --data data/wipe_demos.npz --target opspace --opspace-dims 18 --out checkpoints/op18
$py experiments/cpu/representation/ablation.py --ckpts checkpoints/op6 checkpoints/op12 checkpoints/op18 --labels "pose only (6)" "+wrench (12)" "full (18)" --episodes 30 --out experiments/cpu/representation/results/ablation.json

$py experiments/cpu/representation/collect_data.py --episodes 200 --horizon 250 --control-freq 10 --force-expert --target-force 40 --out data/wipe_force.npz
$py experiments/cpu/representation/train.py --data data/wipe_force.npz --target opspace --out checkpoints/opspace_force
$py experiments/cpu/force/run_force.py --mode scripted --force on --target-force 40 --episodes 10 --json experiments/cpu/force/results/force_tracking.json
$py experiments/cpu/force/run_force.py --mode mlp --force on --ckpt checkpoints/opspace_force --episodes 20 --json experiments/cpu/force/results/force_wipe_on.json
$py experiments/cpu/force/run_force.py --mode mlp --force off --ckpt checkpoints/opspace_force --episodes 20 --json experiments/cpu/force/results/force_wipe_off.json

rm -f experiments/cpu/stiffness/results/stiffness_curves.json
for t in lift can square; do
    curl -fsSL -o data/${t}_ph_low_dim.hdf5 http://downloads.cs.stanford.edu/downloads/rt_benchmark/${t}/ph/low_dim_v141.hdf5
    $py experiments/cpu/stiffness/robomimic.py --mode train --task $t
    for kp in 10 50 150 250 500 1000; do
        $py experiments/cpu/stiffness/robomimic.py --mode eval --task $t --kp $kp --episodes 50
    done
done

$py charts.py
echo "all done"
