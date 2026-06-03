import json, os
import numpy as np

from opspace_vla.env import build_state, PHASE_WIPE, PHASE_APPROACH

INSTRUCTION = "wipe the surface"
PHASE_TEXT = {0: "approach", 1: "contact"}

class MLPForcePolicy:
    def __init__(self, ckpt_dir):
        import torch
        from opspace_vla.models import MLPPolicy
        cfg = json.load(open(os.path.join(ckpt_dir, "config.json")))
        assert cfg["out_dim"] >= 12, "policy must predict the wrench (out_dim >= 12)"
        norm = np.load(os.path.join(ckpt_dir, "norm.npz"))
        self.mean, self.std = norm["state_mean"], norm["state_std"]
        self.model = MLPPolicy(cfg["in_dim"], cfg["out_dim"], hidden=cfg["hidden"])
        self.model.load_state_dict(torch.load(os.path.join(ckpt_dir, "model.pt"),
                                              map_location="cpu"))
        self.model.eval()
        self.torch = torch
        self.with_images = False

    def predict(self, obs, env):
        s = (build_state(obs, env) - self.mean) / self.std
        with self.torch.no_grad():
            return self.model({"state": self.torch.from_numpy(s.astype(np.float32))[None]}).numpy()[0]

class PaliGemmaForcePolicy:
    def __init__(self, ckpt_dir, camera="agentview", device="cuda"):
        import torch
        from transformers import AutoProcessor
        from experiments.paligemma.train_paligemma import PaliGemmaVLA, MODEL_ID
        cfg = json.load(open(os.path.join(ckpt_dir, "config.json")))
        self.device = device
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        vla = PaliGemmaVLA(cfg["out_dim"], lora_dir=os.path.join(ckpt_dir, "lora"))
        vla.head.load_state_dict(torch.load(os.path.join(ckpt_dir, "head.pt"),
                                            map_location=device))
        self.model = vla.to(device).eval()
        self.torch = torch
        self.camera = camera
        self.with_images = True

    def predict(self, obs, env):
        from PIL import Image
        phase = PHASE_WIPE if bool(obs["robot0_contact"]) else PHASE_APPROACH
        img = Image.fromarray(obs[f"{self.camera}_image"])
        prompt = f"{INSTRUCTION} ; phase: {PHASE_TEXT[phase]}"
        enc = self.processor(text=[prompt], images=[img], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            return self.model(**enc).float().cpu().numpy()[0]

def load_mlp_policy(ckpt_dir):
    return MLPForcePolicy(ckpt_dir)

def load_paligemma_policy(ckpt_dir, camera="agentview", device="cuda"):
    return PaliGemmaForcePolicy(ckpt_dir, camera=camera, device=device)
