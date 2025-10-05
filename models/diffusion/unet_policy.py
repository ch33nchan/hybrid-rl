import torch, torch.nn as nn

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, max_steps=10000):
        super().__init__()
        self.dim = dim
        self.max_steps=max_steps
    def forward(self, t):
        device=t.device
        half = self.dim//2
        freqs = torch.exp(-torch.arange(half, device=device) * (torch.log(torch.tensor(10000.0))/ (half-1)))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim%2==1:
            emb = torch.cat([emb, torch.zeros(emb.shape[0],1, device=device)], dim=-1)
        return emb

class SmallUNet1D(nn.Module):
    def __init__(self, seq_dim, model_dim=256, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, model_dim),
            nn.SiLU()
        )
        self.in_proj = nn.Linear(seq_dim, model_dim)
        self.down1 = nn.Sequential(nn.Linear(model_dim, model_dim), nn.SiLU())
        self.mid  = nn.Sequential(nn.Linear(model_dim, model_dim), nn.SiLU())
        self.up1  = nn.Sequential(nn.Linear(model_dim, model_dim), nn.SiLU())
        self.out  = nn.Linear(model_dim, seq_dim)

    def forward(self, x, t):
        temb = self.time_mlp(t)
        h = self.in_proj(x) + temb
        h1 = self.down1(h)
        m  = self.mid(h1)
        h2 = self.up1(m + h1)
        out = self.out(h2)
        return out