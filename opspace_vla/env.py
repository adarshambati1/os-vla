import copy
import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.wipe import DEFAULT_WIPE_CONFIG

PHASE_APPROACH = 0
PHASE_WIPE = 1
N_PHASES = 2

def build_state(obs, env):
    wrench = get_wrench(env)
    phase_onehot = np.zeros(N_PHASES, dtype=np.float32)
    phase_onehot[int(obs.get("_phase", PHASE_APPROACH))] = 1.0
    parts = [
        obs["robot0_eef_pos"],
        obs["robot0_eef_quat"],
        wrench,
        obs["robot0_joint_pos"],
        obs["robot0_joint_vel"],
        np.atleast_1d(obs["wipe_centroid"]),
        np.atleast_1d(obs["gripper_to_wipe_centroid"]),
        np.array([float(obs["proportion_wiped"])], np.float32),
        phase_onehot,
    ]
    return np.concatenate([np.asarray(p, np.float32).ravel() for p in parts])

STATE_DIM = 3 + 4 + 6 + 7 + 7 + 3 + 3 + 1 + N_PHASES

def get_wrench(env):
    r = env.robots[0]
    return np.concatenate([np.asarray(r.ee_force["right"], np.float32),
                           np.asarray(r.ee_torque["right"], np.float32)])

def make_env(controller="osc", with_images=False, horizon=400, seed=None,
             camera="agentview", img_size=128, control_freq=20):
    cfg = load_composite_controller_config(controller="BASIC", robot="Panda")
    if controller == "osc":
        cfg["body_parts"]["right"]["type"] = "OSC_POSE"
        cfg["body_parts"]["right"]["impedance_mode"] = "fixed"
    elif controller == "joint":
        cfg["body_parts"]["right"]["type"] = "JOINT_POSITION"
        cfg["body_parts"]["right"]["output_max"] = [JOINT_DELTA_SCALE] * 7
        cfg["body_parts"]["right"]["output_min"] = [-JOINT_DELTA_SCALE] * 7
    else:
        raise ValueError(controller)

    task_cfg = copy.deepcopy(DEFAULT_WIPE_CONFIG)
    task_cfg["early_terminations"] = False

    env = suite.make(
        env_name="Wipe",
        robots="Panda",
        controller_configs=cfg,
        has_renderer=False,
        has_offscreen_renderer=with_images,
        use_camera_obs=with_images,
        use_object_obs=True,
        camera_names=camera if with_images else None,
        camera_heights=img_size,
        camera_widths=img_size,
        horizon=horizon,
        reward_shaping=True,
        task_config=task_cfg,
        control_freq=control_freq,
        seed=seed,
    )
    return env

OSC_ACTION_DIM = 6
JOINT_ACTION_DIM = 7
OSC_POSE_SLICE = slice(0, 6)
IMPEDANCE_DIM = 6
WRENCH_DIM = 6
OPSPACE_TARGET_DIM = OSC_ACTION_DIM + WRENCH_DIM + IMPEDANCE_DIM
JOINT_DELTA_SCALE = 0.05

def success(env):
    return bool(env._check_success())

def install_hybrid_controller(env, Kfp=0.5, Kfi=0.02, i_clamp=20.0):
    from opspace_vla.controller import WrenchAugmentedOSC
    from robosuite.controllers.parts.arm.osc import OperationalSpaceController
    c = env.robots[0].composite_controller.part_controllers["right"]
    if not isinstance(c, OperationalSpaceController):
        raise TypeError(f"hybrid controller requires an OSC arm controller, got {type(c).__name__}; "
                        f"only valid for the op-space condition (joint uses JOINT_POSITION).")
    c.__class__ = WrenchAugmentedOSC
    c.init_hybrid(Kfp=Kfp, Kfi=Kfi, i_clamp=i_clamp)
    return c
