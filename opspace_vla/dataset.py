import numpy as np
import torch
from torch.utils.data import Dataset
from opspace_vla.env import JOINT_DELTA_SCALE, OSC_ACTION_DIM, WRENCH_DIM, IMPEDANCE_DIM

POSE_SL = slice(0, 6)
WRENCH_SL = slice(6, 12)
IMPED_SL = slice(12, 18)
OPSPACE_DIM = 18

def opspace_targets(d):
    return np.concatenate([d["osc_action"], d["wrench"], d["impedance"]], axis=1).astype(np.float32)

def joint_targets(d):
    dq = (d["next_joint_pos"] - d["joint_pos"]) / JOINT_DELTA_SCALE
    return np.clip(dq, -1.0, 1.0).astype(np.float32)

def episode_split(ep_id, val_frac=0.15, test_frac=0.15, seed=0):
    eps = np.unique(ep_id)
    rng = np.random.RandomState(seed); rng.shuffle(eps)
    n_val = max(1, int(len(eps) * val_frac))
    n_test = max(1, int(len(eps) * test_frac))
    val_eps = set(eps[:n_val].tolist())
    test_eps = set(eps[n_val:n_val + n_test].tolist())
    val_mask = np.array([e in val_eps for e in ep_id])
    test_mask = np.array([e in test_eps for e in ep_id])
    train_mask = ~(val_mask | test_mask)
    return train_mask, val_mask, test_mask

class WipeDataset(Dataset):
    def __init__(self, npz_path, target_kind, split="train", val_frac=0.15, test_frac=0.15,
                 state_mean=None, state_std=None, has_images=False):
        d = np.load(npz_path)
        self.has_images = has_images and ("image" in d.files)
        tr_mask, va_mask, te_mask = episode_split(d["ep_id"], val_frac, test_frac)
        mask = {"train": tr_mask, "val": va_mask, "test": te_mask}[split]

        self.state = d["state"][mask].astype(np.float32)
        self.phase = d["phase"][mask].astype(np.int64)
        self.opspace = opspace_targets(d)[mask]
        self.joint = joint_targets(d)[mask]
        self.target_kind = target_kind
        if self.has_images:
            self.image = d["image"][mask]

        if state_mean is None:
            self.state_mean = self.state.mean(0)
            self.state_std = self.state.std(0) + 1e-6
        else:
            self.state_mean, self.state_std = state_mean, state_std
        self.state_n = (self.state - self.state_mean) / self.state_std

    def target_dim(self):
        return OPSPACE_DIM if self.target_kind == "opspace" else 7

    def __len__(self):
        return len(self.state)

    def __getitem__(self, i):
        y = self.opspace[i] if self.target_kind == "opspace" else self.joint[i]
        item = {
            "state": torch.from_numpy(self.state_n[i]),
            "target": torch.from_numpy(y),
            "phase": int(self.phase[i]),
        }
        if self.has_images:
            img = torch.from_numpy(self.image[i]).permute(2, 0, 1).float() / 255.0
            item["image"] = img
        return item

def opspace_loss_weights():
    w = np.ones(OPSPACE_DIM, np.float32)
    w[WRENCH_SL] = 0.1
    w[IMPED_SL] = 0.01
    return torch.from_numpy(w)
