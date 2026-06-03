import torch
import torch.nn as nn

class MLPPolicy(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, n_layers=3, p_drop=0.0):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                       nn.Dropout(p_drop)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, batch):
        return self.net(batch["state"])

class VisionPolicy(nn.Module):
    def __init__(self, out_dim, proprio_dim=13, hidden=256, p_drop=0.0):
        super().__init__()
        import torchvision
        enc = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.feat_dim = enc.fc.in_features
        enc.fc = nn.Identity()
        self.encoder = enc
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.proprio_dim = proprio_dim
        self.head = nn.Sequential(
            nn.Linear(self.feat_dim + proprio_dim, hidden), nn.LayerNorm(hidden),
            nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, batch):
        x = (batch["image"] - self.mean) / self.std
        with torch.no_grad():
            f = self.encoder(x)
        proprio = batch["state"][:, : self.proprio_dim]
        return self.head(torch.cat([f, proprio], dim=1))
