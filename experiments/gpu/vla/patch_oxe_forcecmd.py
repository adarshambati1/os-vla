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
    '    "libero_forcecmd": {\n'
    '        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},\n'
    '        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},\n'
    '        "state_obs_keys": ["EEF_state", "gripper_state", "wrench"],\n'
    '        "state_encoding": StateEncoding.POS_EULER,\n'
    '        "action_encoding": ActionEncoding.EEF_POS,\n'
    '    },\n'
)
insert_before(f"{OXE}/configs.py", '    "libero_spatial_no_noops": {', cfg, '"libero_forcecmd"')

fn = (
    'def libero_forcecmd_transform(trajectory):\n'
    '    gripper_action = trajectory["action"][:, -1:]\n'
    '    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))\n'
    '    trajectory["action"] = tf.concat([trajectory["action"][:, :7], gripper_action], axis=1)\n'
    '    s = trajectory["observation"]["state"]\n'
    '    trajectory["observation"]["EEF_state"] = s[:, :6]\n'
    '    trajectory["observation"]["gripper_state"] = s[:, 6:8]\n'
    '    trajectory["observation"]["wrench"] = s[:, 8:14]\n'
    '    return trajectory\n\n\n'
)
insert_before(f"{OXE}/transforms.py", "def libero_dataset_transform(", fn, "def libero_forcecmd_transform")
insert_before(f"{OXE}/transforms.py", '    "libero_spatial_no_noops": libero_dataset_transform,',
              '    "libero_forcecmd": libero_forcecmd_transform,\n', '"libero_forcecmd":')
insert_before(f"{OXE}/mixtures.py", '    "libero_spatial_no_noops": [',
              '    "libero_forcecmd": [\n        ("libero_forcecmd", 1.0),\n    ],\n', '"libero_forcecmd"')

mat = f"{OXE}/materialize.py"
m = open(mat).read()
if 'dataset_name == "libero_forcecmd"' not in m:
    anchor = '    dataset_kwargs["action_proprio_normalization_type"] = action_proprio_normalization_type'
    override = (
        '    if dataset_name == "libero_forcecmd":\n'
        '        dataset_kwargs["absolute_action_mask"] = [False] * 6 + [True, True]\n'
        '        dataset_kwargs["action_normalization_mask"] = [True] * 6 + [True, False]\n'
    )
    m = m.replace(anchor, override + anchor, 1)
    open(mat, "w").write(m)
    print(f"{mat}: added libero_forcecmd mask override")
else:
    print(f"{mat}: already patched")
