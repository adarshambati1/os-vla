import argparse, os, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from opspace_vla.dataset import opspace_targets, joint_targets, opspace_loss_weights, episode_split, OPSPACE_DIM

PHASE_TEXT = {0: "approach", 1: "contact"}
INSTRUCTION = "wipe the surface"
MODEL_ID = "google/paligemma-3b-pt-224"

class PGImageDataset(Dataset):
    def __init__(self, npz_path, target_kind, split="train", val_frac=0.15, test_frac=0.15):
        d = np.load(npz_path)
        assert "image" in d.files, "collect with --with-images first"
        tr, va, te = episode_split(d["ep_id"], val_frac, test_frac)
        m = {"train": tr, "val": va, "test": te}[split]
        self.image = d["image"][m]
        self.phase = d["phase"][m].astype(np.int64)
        self.y = (opspace_targets(d) if target_kind == "opspace" else joint_targets(d))[m]
        self.target_kind = target_kind

    def __len__(self):
        return len(self.image)

    def __getitem__(self, i):
        prompt = f"{INSTRUCTION} ; phase: {PHASE_TEXT[int(self.phase[i])]}"
        return {"image": Image.fromarray(self.image[i]),
                "prompt": prompt,
                "target": torch.tensor(self.y[i], dtype=torch.float32)}

class PaliGemmaVLA(nn.Module):
    def __init__(self, out_dim, load_4bit=False, lora_dir=None):
        super().__init__()
        from transformers import PaliGemmaForConditionalGeneration
        from peft import LoraConfig, get_peft_model, PeftModel
        kw = {}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        else:
            kw["torch_dtype"] = torch.bfloat16
        base = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_ID, **kw)
        for p in base.parameters():
            p.requires_grad = False
        if lora_dir is not None:
            self.base = PeftModel.from_pretrained(base, lora_dir)
        else:
            lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                              task_type="FEATURE_EXTRACTION")
            self.base = get_peft_model(base, lora)
        hidden = base.config.text_config.hidden_size
        self.head = nn.Sequential(nn.Linear(hidden, 256), nn.GELU(),
                                  nn.Linear(256, out_dim))

    def forward(self, **inputs):
        out = self.base(**inputs, output_hidden_states=True)
        h = out.hidden_states[-1]
        mask = inputs["attention_mask"].unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled.float())

def collate(processor):
    def fn(batch):
        imgs = [b["image"] for b in batch]
        texts = [b["prompt"] for b in batch]
        enc = processor(text=texts, images=imgs, return_tensors="pt",
                        padding="longest")
        enc["labels_reg"] = torch.stack([b["target"] for b in batch])
        return enc
    return fn

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--target", choices=["opspace", "joint"], required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--load-4bit", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    assert torch.cuda.is_available(), "this needs a GPU"
    device = "cuda"
    os.makedirs(args.out, exist_ok=True)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    out_dim = OPSPACE_DIM if args.target == "opspace" else 7
    tr = PGImageDataset(args.data, args.target, split="train")
    va = PGImageDataset(args.data, args.target, split="val")
    trL = DataLoader(tr, batch_size=args.batch, shuffle=True, collate_fn=collate(processor))
    vaL = DataLoader(va, batch_size=args.batch, shuffle=False, collate_fn=collate(processor))

    model = PaliGemmaVLA(out_dim, load_4bit=args.load_4bit).to(device)
    if args.target == "opspace":
        base = opspace_loss_weights().clone()
        t = tr.y.astype(np.float32)
        for sl in (slice(0, 6), slice(6, 12), slice(12, 18)):
            base[sl] = base[sl] / (float(np.mean(t[:, sl] ** 2)) + 1e-8)
        w = base.to(device)
    else:
        w = torch.ones(7, device=device)
    params = [pp for pp in model.parameters() if pp.requires_grad]
    print(f"trainable params: {sum(pp.numel() for pp in params)/1e6:.1f}M")
    opt = torch.optim.AdamW(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    def step(loader, train):
        model.train(train); tot = n = 0
        for enc in loader:
            y = enc.pop("labels_reg").to(device)
            enc = {k: v.to(device) for k, v in enc.items()}
            pred = model(**enc)
            loss = (w * (pred - y) ** 2).mean()
            if train:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            tot += loss.item() * len(y); n += len(y)
        return tot / max(n, 1)

    hist = {"train": [], "val": []}; best = float("inf")
    for ep in range(args.epochs):
        tl = step(trL, True)
        with torch.no_grad():
            vl = step(vaL, False)
        sched.step(); hist["train"].append(tl); hist["val"].append(vl)
        print(f"ep {ep:3d}  train {tl:.4f}  val {vl:.4f}")
        if vl < best:
            best = vl
            model.base.save_pretrained(os.path.join(args.out, "lora"))
            torch.save(model.head.state_dict(), os.path.join(args.out, "head.pt"))
    json.dump({**vars(args), "out_dim": out_dim, "best_val": best},
              open(os.path.join(args.out, "config.json"), "w"), indent=2)
    json.dump(hist, open(os.path.join(args.out, "history.json"), "w"))
    print(f"done. best val {best:.4f}")

if __name__ == "__main__":
    main()
