import torch

def fit(model, train_loader, val_loader, loss_fn, epochs, lr, save_path, device):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hist = {"train": [], "val": []}
    best = float("inf")
    for ep in range(epochs):
        model.train()
        tot = n = 0
        for b in train_loader:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            loss = loss_fn(model(b), b["target"])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            tot += loss.item() * len(b["target"]); n += len(b["target"])
        sched.step()
        model.eval()
        vt = vn = 0
        with torch.no_grad():
            for b in val_loader:
                b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
                l = loss_fn(model(b), b["target"])
                vt += l.item() * len(b["target"]); vn += len(b["target"])
        tl, vl = tot / max(n, 1), vt / max(vn, 1)
        hist["train"].append(tl); hist["val"].append(vl)
        if vl < best:
            best = vl
            torch.save(model.state_dict(), save_path)
    return best, hist
