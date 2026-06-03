import math
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        ang = t[:, None].float() * freqs[None, :]
        return torch.cat([ang.sin(), ang.cos()], dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.cond = nn.Linear(cond_dim, dim)
        self.act = nn.Mish()

    def forward(self, x, cond):
        h = self.norm(x)
        h = self.act(self.lin1(h) + self.cond(cond))
        h = self.lin2(h)
        return x + h

class NoisePredictor(nn.Module):
    def __init__(self, action_dim, horizon, obs_dim, time_dim=128, hidden=1024, n_blocks=4):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.Mish(),
            nn.Linear(time_dim, time_dim),
        )
        flat = action_dim * horizon
        cond_dim = time_dim + obs_dim
        self.inp = nn.Linear(flat, hidden)
        self.blocks = nn.ModuleList([ResidualBlock(hidden, cond_dim) for _ in range(n_blocks)])
        self.out = nn.Linear(hidden, flat)

    def forward(self, noisy, t, obs_cond):
        b = noisy.shape[0]
        x = self.inp(noisy.reshape(b, -1))
        cond = torch.cat([self.time_mlp(t), obs_cond], dim=-1)
        for blk in self.blocks:
            x = blk(x, cond)
        return self.out(x).reshape(b, self.horizon, self.action_dim)

class DiffusionPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, horizon, n_train_steps=100, hidden=1024, n_blocks=4):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.net = NoisePredictor(action_dim, horizon, obs_dim, hidden=hidden, n_blocks=n_blocks)
        self.scheduler = DDPMScheduler(
            num_train_timesteps=n_train_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def loss(self, obs_cond, actions):
        b = actions.shape[0]
        noise = torch.randn_like(actions)
        t = torch.randint(0, self.scheduler.config.num_train_timesteps, (b,), device=actions.device)
        noisy = self.scheduler.add_noise(actions, noise, t)
        pred = self.net(noisy, t, obs_cond)
        return ((pred - noise) ** 2).mean()

    @torch.no_grad()
    def sample(self, obs_cond, n_inference_steps=16):
        b = obs_cond.shape[0]
        device = obs_cond.device
        x = torch.randn(b, self.horizon, self.action_dim, device=device)
        self.scheduler.set_timesteps(n_inference_steps, device=device)
        for t in self.scheduler.timesteps:
            tb = torch.full((b,), int(t), device=device, dtype=torch.long)
            pred = self.net(x, tb, obs_cond)
            x = self.scheduler.step(pred, t, x).prev_sample
        return x

class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)
