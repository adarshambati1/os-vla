import argparse, json, os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from opspace_vla.robomimic import TASKS, build_state, load_demos, make_env
from opspace_vla.diffusion import DiffusionPolicy, EMA
from opspace_vla.device import pick_device

def ckdir(task):
    return f"checkpoints/diff_{task}"

def make_sequences(S, A, ep, horizon):
    obs, acts = [], []
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        s, a = S[idx], A[idx]
        n = len(s)
        for i in range(n):
            chunk = a[i:i + horizon]
            if len(chunk) < horizon:
                pad = np.repeat(chunk[-1:], horizon - len(chunk), axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)
            obs.append(s[i]); acts.append(chunk)
    return np.asarray(obs, np.float32), np.asarray(acts, np.float32)

class SeqDataset(Dataset):
    def __init__(self, obs, acts):
        self.obs = torch.tensor(obs)
        self.acts = torch.tensor(acts)
    def __len__(self):
        return len(self.obs)
    def __getitem__(self, i):
        return self.obs[i], self.acts[i]

def normalize_actions(A, amin, amax):
    return 2.0 * (A - amin) / (amax - amin) - 1.0

def unnormalize_actions(A, amin, amax):
    return (A + 1.0) / 2.0 * (amax - amin) + amin

def train(task, horizon=16, epochs=400, patience=30, hidden=1024, n_blocks=4, device=None):
    dev = pick_device(device)
    S, A, ep = load_demos(task)
    rng = np.random.RandomState(0); eps = np.unique(ep); rng.shuffle(eps)
    vset = set(eps[:max(1, int(0.15 * len(eps)))].tolist())
    vm = np.array([e in vset for e in ep]); tm = ~vm

    smean, sstd = S[tm].mean(0), S[tm].std(0) + 1e-6
    amin, amax = A[tm].min(0), A[tm].max(0); amax = np.where(amax > amin, amax, amin + 1e-6)

    Sn = (S - smean) / sstd
    An = normalize_actions(A, amin, amax)
    tr_obs, tr_act = make_sequences(Sn[tm], An[tm], ep[tm], horizon)
    va_obs, va_act = make_sequences(Sn[vm], An[vm], ep[vm], horizon)

    trL = DataLoader(SeqDataset(tr_obs, tr_act), batch_size=256, shuffle=True, drop_last=True)
    vaL = DataLoader(SeqDataset(va_obs, va_act), batch_size=256, shuffle=False)

    model = DiffusionPolicy(S.shape[1], A.shape[1], horizon, hidden=hidden, n_blocks=n_blocks).to(dev)
    ema = EMA(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    out = ckdir(task); os.makedirs(out, exist_ok=True)
    ema_model = DiffusionPolicy(S.shape[1], A.shape[1], horizon, hidden=hidden, n_blocks=n_blocks).to(dev)
    best, since = float("inf"), 0
    for epoch in range(epochs):
        model.train()
        for obs, acts in trL:
            obs, acts = obs.to(dev), acts.to(dev)
            loss = model.loss(obs, acts)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            ema.update(model)
        sched.step()

        ema.copy_to(ema_model); ema_model.eval()
        vt = vn = 0
        with torch.no_grad():
            for obs, acts in vaL:
                obs, acts = obs.to(dev), acts.to(dev)
                l = ema_model.loss(obs, acts)
                vt += l.item() * len(obs); vn += len(obs)
        vl = vt / max(vn, 1)
        if vl < best - 1e-5:
            best, since = vl, 0
            torch.save(ema.shadow, os.path.join(out, "model.pt"))
        else:
            since += 1
        if epoch % 10 == 0 or since == 0:
            print(f"[{task}] epoch {epoch}  val {vl:.4f}  best {best:.4f}", flush=True)
        if since >= patience:
            print(f"[{task}] early stop at epoch {epoch}", flush=True)
            break

    np.savez(os.path.join(out, "norm.npz"),
             state_mean=smean, state_std=sstd, action_min=amin, action_max=amax)
    json.dump({"in_dim": int(S.shape[1]), "out_dim": int(A.shape[1]), "horizon": horizon,
               "hidden": hidden, "n_blocks": n_blocks, "best_val": best},
              open(os.path.join(out, "config.json"), "w"), indent=2)
    print(f"[{task}] done. best val {best:.4f}. saved {out}")

def load_policy(task, device):
    out = ckdir(task)
    cfg = json.load(open(os.path.join(out, "config.json")))
    norm = np.load(os.path.join(out, "norm.npz"))
    model = DiffusionPolicy(cfg["in_dim"], cfg["out_dim"], cfg["horizon"],
                            hidden=cfg["hidden"], n_blocks=cfg["n_blocks"]).to(device)
    model.load_state_dict(torch.load(os.path.join(out, "model.pt"), map_location=device))
    model.eval()
    return model, cfg, norm

def evaluate(task, kp, episodes, horizon, act_horizon=8, n_inference=16, batch=50, device=None):
    dev = pick_device(device)
    model, cfg, norm = load_policy(task, dev)
    smean, sstd = norm["state_mean"], norm["state_std"]
    amin, amax = norm["action_min"], norm["action_max"]

    succ = []
    start = 0
    while start < episodes:
        b = min(batch, episodes - start)
        envs = [make_env(task, seed=1000 + start + j, horizon=horizon, kp=kp) for j in range(b)]
        obs = [e.reset() for e in envs]
        done = [False] * b
        plan = [[] for _ in range(b)]
        for _ in range(horizon):
            need = [j for j in range(b) if not done[j] and not plan[j]]
            if need:
                states = np.stack([(build_state(obs[j], task) - smean) / sstd for j in need])
                oc = torch.tensor(states, dtype=torch.float32, device=dev)
                with torch.no_grad():
                    chunks = model.sample(oc, n_inference).cpu().numpy()
                chunks = unnormalize_actions(chunks, amin, amax)
                for k, j in enumerate(need):
                    plan[j] = list(chunks[k][:act_horizon])
            for j in range(b):
                if done[j]:
                    continue
                a = np.clip(plan[j].pop(0), -1, 1)
                o, _, d, _ = envs[j].step(a)
                obs[j] = o
                if d:
                    done[j] = True
            if all(done):
                break
        for j in range(b):
            succ.append(bool(envs[j]._check_success()))
        start += b

    sr = float(np.mean(succ))
    print(f"[{task}] kp={kp:>4.0f}  success {sr:.1%}  over {episodes} ep", flush=True)
    rd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    cf = os.path.join(rd, "diffusion_stiffness_curves.json")
    cur = json.load(open(cf)) if os.path.exists(cf) else {}
    cur.setdefault(task, {})[str(int(kp))] = sr
    os.makedirs(rd, exist_ok=True)
    json.dump(cur, open(cf, "w"), indent=2)
    return sr

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "eval"], required=True)
    p.add_argument("--task", required=True, choices=list(TASKS))
    p.add_argument("--kp", type=float, default=150.0)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--pred-horizon", type=int, default=16)
    p.add_argument("--act-horizon", type=int, default=8)
    p.add_argument("--n-inference", type=int, default=16)
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    if args.mode == "train":
        train(args.task, horizon=args.pred_horizon, epochs=args.epochs, device=args.device)
    else:
        evaluate(args.task, args.kp, args.episodes, args.horizon,
                 act_horizon=args.act_horizon, n_inference=args.n_inference,
                 batch=args.batch, device=args.device)

if __name__ == "__main__":
    main()
