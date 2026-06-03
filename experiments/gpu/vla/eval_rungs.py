import argparse
import json
import os
import sys

import numpy as np
import torch
import imageio

_torch_load = torch.load
def _load_full(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)
torch.load = _load_full

sys.path.append("/workspace/openvla-oft")
sys.path.append("/workspace/openvla-oft/experiments/robot")

_RUNG_ACTION_DIM = {"osc": 7, "forcecmd": 8, "forcewrench": 13, "joint": 8}
_rung_arg = None
for _i, _a in enumerate(sys.argv):
    if _a == "--rung" and _i + 1 < len(sys.argv):
        _rung_arg = sys.argv[_i + 1]
if _rung_arg in _RUNG_ACTION_DIM:
    import prismatic.vla.constants as _C
    _C.ACTION_DIM = _RUNG_ACTION_DIM[_rung_arg]
    _C.PROPRIO_DIM = 14 if _rung_arg in ("forcecmd", "forcewrench") else 8

from collections import deque

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from experiments.robot.libero.libero_utils import get_libero_image, quat2axisangle
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

from hybrid_osc14 import WrenchAugmentedOSC14


TASKS = [
    "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet",
    "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet",
    "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE7_open_the_microwave",
    "KITCHEN_SCENE6_close_the_microwave",
    "KITCHEN_SCENE3_turn_on_the_stove",
    "KITCHEN_SCENE8_turn_off_the_stove",
]

MAX_STEPS = 400
NUM_WAIT = 10
FORCE_CLIP = np.array([100.0, 100.0, 100.0, 20.0, 20.0, 20.0])
FORCE_FAIL_NEWTONS = 80.0

WITH_WRENCH = ("forcecmd", "forcewrench")


class Config:
    def __init__(self, checkpoint, rung):
        self.model_family = "openvla"
        self.pretrained_checkpoint = checkpoint
        self.use_l1_regression = True
        self.use_diffusion = False
        self.num_diffusion_steps_train = 50
        self.num_diffusion_steps_inference = 50
        self.use_film = False
        self.num_images_in_input = 1
        self.use_proprio = True
        self.center_crop = True
        self.num_open_loop_steps = NUM_ACTIONS_CHUNK
        self.lora_rank = 32
        self.load_in_8bit = False
        self.load_in_4bit = False
        self.unnorm_key = f"libero_{rung}"


def measured_wrench(robot):
    raw = np.concatenate([np.asarray(robot.ee_force), np.asarray(robot.ee_torque)])
    return np.clip(-raw, -FORCE_CLIP, FORCE_CLIP)


def make_env(task, rung):
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": bddl, "camera_heights": 256, "camera_widths": 256}
    if rung == "joint":
        env_args["controller"] = "JOINT_POSITION"
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env


def build_state(obs, rung, robot):
    ee = np.concatenate(
        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    )
    if rung in WITH_WRENCH:
        return np.concatenate((ee, measured_wrench(robot)))
    return ee


def setup_force_controller(robot, force_target, force_axes):
    controller = robot.controller
    if not isinstance(controller, WrenchAugmentedOSC14):
        controller.__class__ = WrenchAugmentedOSC14
    controller.init_hybrid()
    controller.set_force_target(force_target, axes=force_axes)


def run_episode(cfg, env, rung, task_label, model, resize_size, processor, action_head,
                proprio_projector, force_target, force_axes, initial_state, video_path=None):
    env.reset()
    obs = env.set_init_state(initial_state)
    robot = env.env.robots[0]
    if rung in WITH_WRENCH:
        setup_force_controller(robot, force_target, force_axes)

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    success = False
    solve_step = None
    peak_force = 0.0
    force_sum = 0.0
    force_count = 0
    commanded = []
    tracking_err = []
    action_mags = []
    frames = []
    dummy_action = ([0.0] * 7 + [-1.0]) if rung == "joint" else ([0.0] * 6 + [-1.0])

    t = 0
    while t < MAX_STEPS + NUM_WAIT:
        if t < NUM_WAIT:
            obs, reward, done, info = env.step(dummy_action)
            t += 1
            continue

        wrench = measured_wrench(robot)
        contact = float(np.max(np.abs(wrench[:3])))
        peak_force = max(peak_force, contact)
        force_sum += contact
        force_count += 1
        if rung in WITH_WRENCH:
            robot.controller.set_measured_wrench(wrench)

        frame = get_libero_image(obs)
        if video_path is not None:
            frames.append(frame)
        observation = {
            "full_image": resize_image_for_policy(frame, resize_size),
            "state": build_state(obs, rung, robot),
        }
        if len(action_queue) == 0:
            action_queue.extend(get_action(
                cfg, model, observation, task_label,
                processor=processor, action_head=action_head,
                proprio_projector=proprio_projector, use_film=cfg.use_film,
            ))
        action = action_queue.popleft()
        action_mags.append(float(np.mean(np.abs(np.asarray(action)))))

        if rung == "joint":
            action = action.copy()
            action[:7] = np.clip(action[:7] / 0.05, -1.0, 1.0)
        elif rung == "forcecmd":
            fz_cmd = float(action[6])
            commanded.append(fz_cmd)
            tracking_err.append(abs(fz_cmd - float(wrench[2])))
            robot.controller.set_force_target([0.0, 0.0, fz_cmd, 0.0, 0.0, 0.0], axes=(2,))
            action = np.array([action[0], action[1], action[2], action[3], action[4], action[5], action[7]])
        elif rung == "forcewrench":
            w_cmd = np.asarray(action[6:12], float)
            commanded.append(float(w_cmd[2]))
            tracking_err.append(abs(float(w_cmd[2]) - float(wrench[2])))
            robot.controller.set_feedforward_wrench(w_cmd)
            action = np.array([action[0], action[1], action[2], action[3], action[4], action[5], action[12]])

        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)

        obs, reward, done, info = env.step(action.tolist())
        t += 1
        if done:
            success = True
            solve_step = t - NUM_WAIT
            break

    mean_force = force_sum / force_count if force_count else 0.0
    stats = {
        "success": success,
        "solve_step": solve_step,
        "peak_force": peak_force,
        "mean_force": mean_force,
        "force_fail": peak_force > FORCE_FAIL_NEWTONS,
        "mean_commanded_fz": float(np.mean(commanded)) if commanded else None,
        "mean_force_tracking_err": float(np.mean(tracking_err)) if tracking_err else None,
        "mean_action_mag": float(np.mean(action_mags)) if action_mags else None,
    }
    if video_path is not None and frames:
        imageio.mimsave(video_path, frames, fps=20)
        print(f"saved video {video_path} ({len(frames)} frames, success={success})", flush=True)
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--rung", required=True, choices=["osc", "forcecmd", "forcewrench", "joint"])
    p.add_argument("--num_trials", type=int, default=50)
    p.add_argument("--tasks", default=None, help="comma-separated task indices; default all 10")
    p.add_argument("--out", default=None)
    p.add_argument("--save_video", action="store_true", help="save mp4 of the first 2 trials of each task")
    p.add_argument("--video_dir", default="/workspace/videos")
    p.add_argument("--max_steps", type=int, default=400, help="rollout step budget per episode")
    args = p.parse_args()
    global MAX_STEPS
    MAX_STEPS = args.max_steps
    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)

    task_names = TASKS if args.tasks is None else [TASKS[int(i)] for i in args.tasks.split(",")]

    set_seed_everywhere(7)
    cfg = Config(args.checkpoint, args.rung)
    proprio_dim = 14 if args.rung in WITH_WRENCH else 8
    force_target = np.zeros(6)
    force_axes = (2,) if args.rung == "forcecmd" else ()

    model = get_model(cfg)
    if cfg.unnorm_key not in model.norm_stats:
        keys = list(model.norm_stats.keys())
        assert len(keys) == 1, f"unnorm_key {cfg.unnorm_key} not found; choices {keys}"
        cfg.unnorm_key = keys[0]
        print(f"using unnorm_key {cfg.unnorm_key}")

    processor = get_processor(cfg)
    action_head = get_action_head(cfg, model.llm_dim)
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=proprio_dim)
    resize_size = get_image_resize_size(cfg)

    suite = benchmark.get_benchmark_dict()["libero_90"]()
    name_to_id = {suite.get_task(i).name: i for i in range(suite.n_tasks)}

    results = {}
    for name in task_names:
        task = suite.get_task(name_to_id[name])
        env = make_env(task, args.rung)
        init_states = suite.get_task_init_states(name_to_id[name])
        ep_stats = []
        print(f"[{args.rung}] START {name[:38]} ({args.num_trials} trials)", flush=True)
        for ep in range(args.num_trials):
            vp = None
            if args.save_video and ep < 2:
                vp = f"{args.video_dir}/{args.rung}_{name[:25]}_ep{ep}.mp4"
            ep_stats.append(run_episode(
                cfg, env, args.rung, task.language, model, resize_size,
                processor, action_head, proprio_projector, force_target, force_axes,
                init_states[ep % len(init_states)], video_path=vp,
            ))
            if (ep + 1) % 5 == 0:
                running = sum(s["success"] for s in ep_stats)
                print(f"[{args.rung}] {name[:30]} ep {ep + 1}/{args.num_trials} succ={running}", flush=True)
        env.close()

        successes = [s for s in ep_stats if s["success"]]
        solve = [s["solve_step"] for s in successes]
        commanded = [s["mean_commanded_fz"] for s in ep_stats if s["mean_commanded_fz"] is not None]
        track = [s["mean_force_tracking_err"] for s in ep_stats if s["mean_force_tracking_err"] is not None]
        results[name] = {
            "success_rate": len(successes) / args.num_trials,
            "successes": len(successes),
            "trials": args.num_trials,
            "mean_steps_to_success": float(np.mean(solve)) if solve else None,
            "mean_peak_force": float(np.mean([s["peak_force"] for s in ep_stats])),
            "max_peak_force": float(np.max([s["peak_force"] for s in ep_stats])),
            "mean_force": float(np.mean([s["mean_force"] for s in ep_stats])),
            "force_failure_rate": float(np.mean([s["force_fail"] for s in ep_stats])),
            "mean_commanded_fz": float(np.mean(commanded)) if commanded else None,
            "mean_force_tracking_err": float(np.mean(track)) if track else None,
            "mean_action_mag": float(np.mean([s["mean_action_mag"] for s in ep_stats if s["mean_action_mag"] is not None])),
        }
        r = results[name]
        steps_str = f"{r['mean_steps_to_success']:.0f}" if r["mean_steps_to_success"] is not None else "n/a"
        print(f"{name:55s} succ {r['success_rate']:.2f}  steps {steps_str}  "
              f"peakF {r['mean_peak_force']:.1f}N  failF {r['force_failure_rate']:.2f}")

    def avg(key, sub=False):
        vals = [r[key] for r in results.values() if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    out = {
        "rung": args.rung,
        "checkpoint": args.checkpoint,
        "overall_success_rate": float(np.mean([r["success_rate"] for r in results.values()])),
        "overall_mean_steps_to_success": avg("mean_steps_to_success"),
        "overall_mean_peak_force": avg("mean_peak_force"),
        "overall_mean_force": avg("mean_force"),
        "overall_force_failure_rate": avg("force_failure_rate"),
        "overall_mean_commanded_fz": avg("mean_commanded_fz"),
        "overall_mean_force_tracking_err": avg("mean_force_tracking_err"),
        "overall_mean_action_mag": avg("mean_action_mag"),
        "per_task": results,
    }
    path = args.out or f"/workspace/eval_{args.rung}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{args.rung.upper()} overall success {out['overall_success_rate']:.3f}  ->  {path}")


if __name__ == "__main__":
    main()
