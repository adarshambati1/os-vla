import re
import glob
from typing import Iterator, Tuple, Any

import numpy as np
import h5py
import tensorflow_datasets as tfds

DSDIR = "/workspace/LIBERO/libero/datasets/libero_90"
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

def language_of(name):
    return re.sub(r"^KITCHEN_SCENE\d+_", "", name).replace("_", " ")

class LiberoOsc(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Heavy-contact LIBERO-90 tasks, OSC actions."}

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(shape=(128, 128, 3), dtype=np.uint8),
                        "state": tfds.features.Tensor(shape=(8,), dtype=np.float32),
                    }),
                    "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(),
                }),
            }))

    def _split_generators(self, dl_manager):
        return {"train": self._generate_examples()}

    def _generate_examples(self) -> Iterator[Tuple[str, Any]]:
        for name in TASKS:
            lang = language_of(name)
            path = f"{DSDIR}/{name}_demo.hdf5"
            f = h5py.File(path, "r")
            data = f["data"]
            for dn in data.keys():
                obs = data[dn]["obs"]
                imgs = obs["agentview_rgb"][:]
                acts = np.asarray(data[dn]["actions"][:], np.float32)
                state = np.concatenate([obs["ee_pos"][:], obs["ee_ori"][:], obs["gripper_states"][:]], axis=1).astype(np.float32)
                n = len(acts)
                steps = []
                for t in range(n):
                    steps.append({
                        "observation": {
                            "image": np.rot90(imgs[t], 2).copy(),
                            "state": state[t],
                        },
                        "action": acts[t],
                        "discount": 1.0,
                        "reward": float(t == n - 1),
                        "is_first": t == 0,
                        "is_last": t == n - 1,
                        "is_terminal": t == n - 1,
                        "language_instruction": lang,
                    })
                yield f"{name}_{dn}", {"steps": steps, "episode_metadata": {"file_path": path}}
            f.close()
