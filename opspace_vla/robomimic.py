import numpy as np

TASKS = {
    "lift":   dict(env="Lift", data="data/lift_ph_low_dim.hdf5",
                   objkeys=["cube_pos", "cube_quat", "gripper_to_cube_pos"]),
    "can":    dict(env="PickPlaceCan", data="data/can_ph_low_dim.hdf5",
                   objkeys=["Can_pos", "Can_quat", "Can_to_robot0_eef_pos", "Can_to_robot0_eef_quat"]),
    "square": dict(env="NutAssemblySquare", data="data/square_ph_low_dim.hdf5",
                   objkeys=["SquareNut_pos", "SquareNut_quat", "SquareNut_to_robot0_eef_pos", "SquareNut_to_robot0_eef_quat"]),
    "tool_hang": dict(env="ToolHang", data="data/tool_hang_ph_low_dim.hdf5",
                   objkeys=["base_pos", "base_quat", "base_to_robot0_eef_pos", "base_to_robot0_eef_quat",
                            "frame_pos", "frame_quat", "frame_to_robot0_eef_pos", "frame_to_robot0_eef_quat",
                            "tool_pos", "tool_quat", "tool_to_robot0_eef_pos", "tool_to_robot0_eef_quat",
                            "frame_is_assembled", "tool_on_frame"]),
}

def obj_from_env(obs, task):
    if task == "lift":
        rel = np.asarray(obs["robot0_eef_pos"], np.float32) - np.asarray(obs["cube_pos"], np.float32)
        return np.concatenate([np.asarray(obs["cube_pos"], np.float32).ravel(),
                               np.asarray(obs["cube_quat"], np.float32).ravel(), rel])
    return np.concatenate([np.asarray(obs[k], np.float32).ravel() for k in TASKS[task]["objkeys"]])

def build_state(obs, task):
    obj = obs["object"] if "object" in obs else obj_from_env(obs, task)
    return np.concatenate([
        np.asarray(obs["robot0_eef_pos"], np.float32).ravel(),
        np.asarray(obs["robot0_eef_quat"], np.float32).ravel(),
        np.asarray(obs["robot0_gripper_qpos"], np.float32).ravel(),
        np.asarray(obj, np.float32).ravel()]).astype(np.float32)

def load_demos(task):
    import h5py
    f = h5py.File(TASKS[task]["data"], "r")
    demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
    S, A, ep = [], [], []
    for i, dn in enumerate(demos):
        o = f["data"][dn]["obs"]; n = f["data"][dn]["actions"].shape[0]
        s = np.concatenate([o["robot0_eef_pos"][:], o["robot0_eef_quat"][:],
                            o["robot0_gripper_qpos"][:], o["object"][:]], axis=1).astype(np.float32)
        S.append(s); A.append(np.asarray(f["data"][dn]["actions"][:], np.float32)); ep.append(np.full(n, i))
    f.close()
    return np.concatenate(S), np.concatenate(A), np.concatenate(ep)

def make_env(task, seed=None, horizon=400, kp=150.0):
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config
    cfg = load_composite_controller_config(controller="BASIC", robot="Panda")
    cfg["body_parts"]["right"]["type"] = "OSC_POSE"
    cfg["body_parts"]["right"]["impedance_mode"] = "fixed"
    cfg["body_parts"]["right"]["kp"] = kp
    return suite.make(TASKS[task]["env"], robots="Panda", controller_configs=cfg,
                      has_renderer=False, use_camera_obs=False, use_object_obs=True,
                      horizon=horizon, control_freq=20, seed=seed)

def ckdir(task):
    return f"checkpoints/rmbc_{task}"
