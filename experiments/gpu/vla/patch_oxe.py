import sys

OXE = "/workspace/openvla-oft/prismatic/vla/datasets/rlds/oxe"
name = sys.argv[1] if len(sys.argv) > 1 else "libero_osc"

def patch(path, anchor, insert):
    with open(path) as f:
        s = f.read()
    if f'"{name}"' in s:
        print(f"{path}: {name} already present, skip")
        return
    if anchor not in s:
        print(f"{path}: ANCHOR NOT FOUND")
        return
    s = s.replace(anchor, insert + anchor, 1)
    with open(path, "w") as f:
        f.write(s)
    print(f"{path}: inserted {name}")

cfg = (
    f'    "{name}": {{\n'
    '        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},\n'
    '        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},\n'
    '        "state_obs_keys": ["EEF_state", "gripper_state"],\n'
    '        "state_encoding": StateEncoding.POS_EULER,\n'
    '        "action_encoding": ActionEncoding.EEF_POS,\n'
    '    },\n'
)
patch(f"{OXE}/configs.py", '    "libero_spatial_no_noops": {', cfg)
patch(f"{OXE}/transforms.py", '    "libero_spatial_no_noops": libero_dataset_transform,',
      f'    "{name}": libero_dataset_transform,\n')
patch(f"{OXE}/mixtures.py", '    "libero_spatial_no_noops": [',
      f'    "{name}": [\n        ("{name}", 1.0),\n    ],\n')
