# SimToolReal Codebase Reference Notes

Survey of `simtoolreal-reference/` (cloned alongside this repo) — what
we'll need to consume their data and Isaac Gym environments.

## 1. Repository Layout

- **assets/** — URDFs (KUKA arm + Sharpa hand variants), object meshes
  for DexToolBench (hammer, marker, eraser, brush, spatula, screwdriver).
- **baselines/** — Kinematic retargeting (SAM2 + HaMeR) and fixed-grasp
  trajectory optimization (Pyroki).
- **deployment/** — Sim2Real ROS nodes (RL policy, goal pose, robot/hand
  control, perception) and Sim2Sim equivalents.
- **dextoolbench/** — Benchmark interface: object defs, task trajectories,
  evaluation, data download.
- **isaacgymenvs/** — Isaac Gym environments. Core class
  `SimToolRealLSTMAsymmetric` in `tasks/simtoolreal/env.py`. RL training
  via `launch_training.py`.
- **recorded_data/** — `RecordedData` dataclass (`core.py`), numpy
  `savez` format, plus loaders/visualizers.
- **rl_games/** — Fork with PPO and SAPG (Stabilized Advantage Policy
  Gradient).

## 2. Rollout Generation & Data Collection

### Entry points

- **Training:** `isaacgymenvs/launch_training.py` → `train.py`. Args:
  `--custom_experiment_name`, `--checkpoint`, `--num_envs`,
  `--num_blocks` (SAPG).
- **Interactive eval:** `dextoolbench/eval_interactive.py` (Viser web UI,
  port 8080). Args: `--config-path`, `--checkpoint-path`, `--port`.
- **Batch eval:** `dextoolbench/run_all_evals.py`.

### Rollout data format

- **Format:** compressed numpy archive (`.npz`), saved via
  `RecordedData.to_file()` (`recorded_data/core.py:209`).
- **Path convention:** `recorded_robot_inputs/<timestamp>/<timestamp>_model_*.npz`.
- **Fields** (`recorded_data/core.py:145-160`):

| Field                              | Shape       | Notes                                   |
| ---------------------------------- | ----------- | --------------------------------------- |
| `robot_root_states_array`          | `(T, 13)`   | xyz, xyzw quat, linvel(3), angvel(3)    |
| `object_root_states_array`         | `(T, 13)`   | same                                    |
| `robot_joint_positions_array`      | `(T, 29)`   | 7 arm + 22 hand                         |
| `time_array`                       | `(T,)`      |                                         |
| `robot_joint_velocities_array`*    | `(T, 29)`   |                                         |
| `robot_joint_pos_targets_array`*   | `(T, 29)`   | commanded targets                       |
| `observations_array`*              | `(T, 133)`  | full asymmetric state                   |
| `actions_array`*                   | `(T, 29)`   | scaled action                           |
| `robot_joint_names`                | list[str]   | `ADJUSTED_JOINT_ORDER`                  |
| `table_root_states_array`*         | `(T, 13)`   |                                         |
| `goal_root_states_array`*          | `(T, 13)`   |                                         |

`*` = optional. Loaded via `RecordedData.from_file(path)`.

## 3. Isaac Gym Environment API

- **Main class:** `SimToolReal` at `isaacgymenvs/tasks/simtoolreal/env.py:88`.
- **Observation dims:** 133 (full state). Asymmetric policy uses ~100.
  Decomposition (`env.py:301-320`): joint q (29), joint qd (29), prev
  targets (29), palm pos+quat (3+4), palm linvel+angvel (3+3), object
  quat (4), object linvel+angvel (3+3), fingertip-rel-palm (15),
  keypoints rel palm (12), keypoints rel goal (12), object scale (3),
  closest keypoint dist (1), closest fingertip dists (5), lifted (1),
  progress/successes/reward (3).
- **Action dims:** 29 hybrid:
  - **arm[0:7]**: relative velocity targets, scaled by `dofSpeedScale=1.5`
    and `armMovingAverage=0.1`, clamped to joint limits
    (`env.py:491-504`).
  - **hand[7:29]**: position targets, scaled to joint limits, smoothed
    via `handMovingAverage=0.1`.
- **Control frequency:** 60 Hz (`controlFrequencyInv=1`, sim dt ~0.0083 s).
- **Episode length:** 600 steps (~10 s).

## 4. Contact Forces & Physics State

- **Fingertip force-torque sensors:** off by default
  (`with_fingertip_force_sensors=False` at `env.py:230`). When on,
  acquired via
  `self.gym.acquire_force_sensor_tensor(self.sim)` and wrapped with
  `gymtorch.wrap_tensor()` (`env.py:440-447`).
- **Optional table FT sensor:** `withTableForceSensor` flag (`env.py:231`).
- **Rigid-body root states:**
  `self.gym.acquire_actor_root_state_tensor(self.sim)` → tensor of shape
  `(N, 13)` per actor (pos 3, quat 4, linvel 3, angvel 3).
- **End-effector / palm:** computed from forward kinematics on link
  `iiwa14_link_7` plus per-finger DPs in
  `observation_action_utils_sharpa.py:507-545` (`_compute_palm_center_pos_and_rot`).
- **Fingertip links:** `right/left_{index,middle,ring,thumb,pinky}_DP`
  (env.py:261-275).

> **Implication for OS-VLA:** to extract per-step contact wrench labels
> we'll have to *enable* the fingertip + (optionally) palm/tool force
> sensors. They're already wired in — flip the flag during data
> collection. Total tool-tip wrench = sum of fingertip wrenches in tool
> frame.

## 5. Recorded Data Structure

- **One `.npz` per episode/recording session.** Filename pattern:
  `<timestamp>_model_*.npz`.
- **Dtype:** `float64` everywhere except joint_names (list[str]).
- **Access:** `RecordedData.from_file(path)` / `.to_file(path)` in
  `recorded_data/core.py:209,227`.

## 6. Tool Primitives (DexToolBench)

24 canonical tasks = 6 categories × 2 object instances × 2 variations.
Defined in `dextoolbench/objects.py` (`NAME_TO_OBJECT`).

| Category    | Instances                       | Tasks                            |
| ----------- | ------------------------------- | -------------------------------- |
| hammer      | claw, mallet                    | swing_down, swing_side           |
| marker      | sharpie, staples                | draw_smile, write_c              |
| eraser      | flat, handle                    | wipe_smile, wipe_c               |
| brush       | red, blue                       | sweep_forward, sweep_right       |
| spatula     | flat, spoon                     | serve_plate, flip_over           |
| screwdriver | long, short                     | spin_vertical, spin_horizontal   |

URDFs/meshes under `assets/urdf/dextoolbench/<category>/<object>/`.
Object scales applied 25× from physical dims (~7-20 cm).

## 7. Pretrained Policies

`download_pretrained_policy.py:89`:
- URL: `https://download.cs.stanford.edu/juno/simtoolreal/pretrained_policy.zip`
- Extracts to `pretrained_policy/` with `config.yaml` and `model.pth`.

Two baseline pipelines shipped:
1. **Kinematic retargeting** — `baselines/visualize_demo_with_hand.py`:
   SAM2 + HaMeR → URDF retargeting → replay.
2. **Fixed-grasp trajopt** — `baselines/visualize_demo_with_hand_trajopt.py`
   (requires Pyroki): grasp via RL policy, then trajopt for the rest.

Both produce `.npz` joint-target trajectories replayable via
`deployment/replay_trajectory.py`.

## 8. README Highlights

- Object-centric RL framework for dexterous tool manipulation, combining
  Isaac Gym + SAPG + Sim2Real (ROS + FoundationPose + SAM).
- Modular Sim2Real stack: policy node, goal-pose node, perception node,
  robot/hand controllers, Viser-based 3D vis.
- DexToolBench: 24 canonical tasks with real RGB-D + object masks +
  poses + reference trajectories. Designed for zero-shot sim-to-real.
- Training: RL Games (SAPG), 8k-24k parallel envs, curriculum via
  tolerance annealing, randomization (forces, torques, object props),
  WandB logging.
- Data utilities: `.npz` recording with full state/obs/actions/targets,
  perception pipelines (SAM2 + FoundationPose) for mesh/pose extraction.

## Key file references

- Env class: `isaacgymenvs/tasks/simtoolreal/env.py:88`
- Obs/action utils: `isaacgymenvs/utils/observation_action_utils_sharpa.py:266` (compute_observation), `:445` (compute_joint_pos_targets), `:507` (palm pose)
- Force sensors: `tasks/simtoolreal/env.py:440`
- RecordedData: `recorded_data/core.py:146,209,227`
- Object catalog: `dextoolbench/objects.py` (`NAME_TO_OBJECT`)
- Task config: `isaacgymenvs/cfg/task/SimToolReal.yaml`
- Sim2Real entry points: `deployment/rl_policy_node.py`,
  `deployment/goal_pose_node.py`, `deployment/visualization_node.py`
- Policy download: `download_pretrained_policy.py:89`

## Resolved questions for OS-VLA

1. **Joint dim** — 29 (7 KUKA + 22 Sharpa), **not** the 16 in the original
   sketch. Update the baseline head accordingly.
2. **EE frame for pose label** — palm-center transform from
   `_compute_palm_center_pos_and_rot`. Convert quat → axis-angle for the
   6D pose target.
3. **Wrench label** — flip on `with_fingertip_force_sensors=True` during
   collection; sum the per-tip wrenches into the tool frame.
4. **Impedance label** — Isaac Gym does not expose stiffness/damping
   directly. Estimate per-window from
   `wrench_t / (delta_pose_{t-k..t})` for stiffness and
   `wrench_t / (delta_vel_{t-k..t})` for damping (regression with a
   small ridge to keep PSD), or treat impedance as a learnable target
   shaped by task phase.
5. **Image source** — DexToolBench ships RGB-D for the 24 task scenes;
   in-sim we'll need to attach an Isaac Gym camera at the same viewpoint.

## Open questions

- How exactly is the per-task instruction encoded? (No language string
  in the `.npz`; we'll have to mint instructions per (category, task).)
- Asymmetric obs vs full obs: which 100D subset is the policy input?
  (Need to read `observation_action_utils_sharpa.py:266` more carefully.)
- Does the camera setup in `eval_interactive.py` match DexToolBench RGB-D
  intrinsics? Determines whether sim images and real images share a
  distribution.
