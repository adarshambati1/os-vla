OXE = "/workspace/openvla-oft/prismatic/vla/datasets/rlds/oxe"

def insert_before(path, anchor, insert, guard):
    with open(path) as f:
        s = f.read()
    if guard in s:
        print(f"{path}: already patched"); return
    if anchor not in s:
        print(f"{path}: ANCHOR NOT FOUND"); return
    s = s.replace(anchor, insert + anchor, 1)
    with open(path, "w") as f:
        f.write(s)
    print(f"{path}: patched")

cfg = (
    '    "libero_joint": {\n'
    '        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},\n'
    '        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},\n'
    '        "state_obs_keys": ["EEF_state", "gripper_state"],\n'
    '        "state_encoding": StateEncoding.POS_EULER,\n'
    '        "action_encoding": ActionEncoding.JOINT_POS,\n'
    '    },\n'
)
insert_before(f"{OXE}/configs.py", '    "libero_spatial_no_noops": {', cfg, '"libero_joint"')

fn = (
    'def libero_joint_transform(trajectory):\n'
    '    gripper_action = trajectory["action"][:, -1:]\n'
    '    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))\n'
    '    trajectory["action"] = tf.concat([trajectory["action"][:, :7], gripper_action], axis=1)\n'
    '    s = trajectory["observation"]["state"]\n'
    '    trajectory["observation"]["EEF_state"] = s[:, :6]\n'
    '    trajectory["observation"]["gripper_state"] = s[:, 6:8]\n'
    '    return trajectory\n\n\n'
)
insert_before(f"{OXE}/transforms.py", "def libero_dataset_transform(", fn, "def libero_joint_transform")
insert_before(f"{OXE}/transforms.py", '    "libero_spatial_no_noops": libero_dataset_transform,',
              '    "libero_joint": libero_joint_transform,\n', '"libero_joint":')
insert_before(f"{OXE}/mixtures.py", '    "libero_spatial_no_noops": [',
              '    "libero_joint": [\n        ("libero_joint", 1.0),\n    ],\n', '"libero_joint"')

mat = f"{OXE}/materialize.py"
m = open(mat).read()
if "is ActionEncoding.JOINT_POS:" not in m:
    m = m.replace("ActionEncoding.JOINT_POS_BIMANUAL]:",
                  "ActionEncoding.JOINT_POS, ActionEncoding.JOINT_POS_BIMANUAL]:")
    anchor = '    elif dataset_kwargs["action_encoding"] is ActionEncoding.EEF_R6:'
    branch = ('    elif dataset_kwargs["action_encoding"] is ActionEncoding.JOINT_POS:\n'
              '        dataset_kwargs["absolute_action_mask"] = [False] * 7 + [True]\n'
              '        dataset_kwargs["action_normalization_mask"] = [True] * 7 + [False]\n')
    m = m.replace(anchor, branch + anchor, 1)
    open(mat, "w").write(m)
    print(f"{mat}: added JOINT_POS branch")
else:
    print(f"{mat}: already patched")
