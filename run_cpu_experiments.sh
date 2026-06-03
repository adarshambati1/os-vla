#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
py="conda run -n opspace-vla python"

$py experiments/representation/collect_data.py --episodes 200 --horizon 250 --control-freq 10 --hybrid --out data/wipe_demos.npz
$py experiments/representation/train.py --data data/wipe_demos.npz --target opspace --out checkpoints/opspace
$py experiments/representation/train.py --data data/wipe_demos.npz --target joint --out checkpoints/joint
$py experiments/representation/eval.py --opspace checkpoints/opspace --joint checkpoints/joint --episodes 30 --hybrid --json results/closedloop.json

$py experiments/representation/collect_data.py --episodes 60 --horizon 250 --control-freq 10 --with-images --img-size 128 --hybrid --out data/wipe_img.npz
$py experiments/representation/train.py --data data/wipe_img.npz --target opspace --vision --device mps --out checkpoints/opspace_vis
$py experiments/representation/train.py --data data/wipe_img.npz --target joint --vision --device mps --out checkpoints/joint_vis
$py experiments/representation/eval.py --opspace checkpoints/opspace_vis --joint checkpoints/joint_vis --episodes 30 --img-size 128 --hybrid --json results/closedloop_vision.json

$py experiments/representation/train.py --data data/wipe_demos.npz --target opspace --opspace-dims 6 --out checkpoints/op6
$py experiments/representation/train.py --data data/wipe_demos.npz --target opspace --opspace-dims 12 --out checkpoints/op12
$py experiments/representation/train.py --data data/wipe_demos.npz --target opspace --opspace-dims 18 --out checkpoints/op18
$py experiments/representation/ablation.py --ckpts checkpoints/op6 checkpoints/op12 checkpoints/op18 --labels "pose only (6)" "+wrench (12)" "full (18)" --episodes 30 --out results/ablation.json

$py experiments/representation/collect_data.py --episodes 200 --horizon 250 --control-freq 10 --force-expert --target-force 40 --out data/wipe_force.npz
$py experiments/representation/train.py --data data/wipe_force.npz --target opspace --out checkpoints/opspace_force
$py experiments/force/run_force.py --mode scripted --force on --target-force 40 --episodes 10 --json results/force_tracking.json
$py experiments/force/run_force.py --mode mlp --force on --ckpt checkpoints/opspace_force --episodes 20 --json results/force_wipe_on.json
$py experiments/force/run_force.py --mode mlp --force off --ckpt checkpoints/opspace_force --episodes 20 --json results/force_wipe_off.json

#exrpereiment to see if different tasks perform better with different kp stiffness
rm -f results/stiffness_curves.json
for t in lift can square; do
    curl -fsSL -o data/${t}_ph_low_dim.hdf5 http://downloads.cs.stanford.edu/downloads/rt_benchmark/${t}/ph/low_dim_v141.hdf5
    $py experiments/stiffness/robomimic.py --mode train --task $t
    for kp in 10 50 150 250 500 1000; do
        $py experiments/stiffness/robomimic.py --mode eval --task $t --kp $kp --episodes 50
    done
done

$py charts.py
echo "all done"
