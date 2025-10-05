import argparse, torch, numpy as np, math, json, random
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from lerobot_dataset.pick_place_mj_builder import MjPickPlaceOfflineDataset
from utils.phase_labeling import label_phase

def cosine_beta_schedule(T, s=0.008):
    steps = T+1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x/T)+s)/(1+s) * math.pi*0.5)**2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:]/alphas_cumprod[:-1])
    return torch.clip(betas, 1e-5, 0.02)

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim=dim
    def forward(self, t):
        device=t.device
        half=self.dim//2
        freqs=torch.exp(-torch.arange(half, device=device)*(torch.log(torch.tensor(10000.))/(half-1)))
        args=t.float().unsqueeze(1)*freqs.unsqueeze(0)
        emb=torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim%2: emb=torch.cat([emb, torch.zeros(emb.shape[0],1,device=device)], dim=-1)
        return emb

class CondDiffMLP(nn.Module):
    def __init__(self, seq_dim, state_dim, phase_dim=6, phase_emb=16, time_dim=128, hidden=512, dropout=0.05):
        super().__init__()
        self.time = nn.Sequential(SinusoidalTimeEmbedding(time_dim), nn.Linear(time_dim, hidden), nn.SiLU())
        self.phase_emb = nn.Embedding(phase_dim, phase_emb)
        in_dim = seq_dim + state_dim + phase_emb
        self.in_lin = nn.Linear(in_dim, hidden)
        self.block1 = nn.Sequential(nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.block2 = nn.Sequential(nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.out = nn.Linear(hidden, seq_dim)
    def forward(self, noisy_seq, state_cond, phase_ids, t, cond_drop_prob=0.0):
        if self.training and cond_drop_prob>0:
            drop_mask = (torch.rand(phase_ids.shape, device=noisy_seq.device) < cond_drop_prob).float().unsqueeze(1)
            state_cond = state_cond * (1 - drop_mask)
        pemb = self.phase_emb(phase_ids)
        x = torch.cat([noisy_seq, state_cond, pemb], dim=-1)
        h = self.in_lin(x)
        temb = self.time(t)
        h = h + temb
        h = self.block1(h)
        h = self.block2(h)
        return self.out(h)

def build_sequences(base, horizon, stride, max_samples=None):
    seqs=[]; cond_states=[]; phases=[]
    N=len(base)
    for i in range(0, N - horizon, stride):
        ep_i, t_i = base.index[i]
        ep_j, t_j = base.index[i+horizon-1]
        if ep_i != ep_j: continue
        acts=[]
        ok=True
        for k in range(horizon):
            ep_k, tk = base.index[i+k]
            if ep_k != ep_i: ok=False; break
            acts.append(base[i+k]["action"])
        if not ok: continue
        seqs.append(np.concatenate(acts, axis=0))
        s0 = base[i]["obs_state"]
        cond_states.append(s0)
        phases.append(label_phase(s0))
        if max_samples and len(seqs)>=max_samples: break
    return (np.array(seqs,dtype=np.float32),
            np.array(cond_states,dtype=np.float32),
            np.array(phases,dtype=np.int64))

def normalize(seqs):
    mean=seqs.mean(0, keepdims=True)
    std =seqs.std(0, keepdims=True)+1e-6
    return (seqs-mean)/std, mean.squeeze(0), std.squeeze(0)

def main(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    base = MjPickPlaceOfflineDataset(a.data_root, use_paraphrase=False)
    seqs, cond_states, phases = build_sequences(base, a.horizon, a.stride, max_samples=a.max_samples)
    seqs_norm, act_mean, act_std = normalize(seqs)
    seq_dim = seqs_norm.shape[1]; state_dim = cond_states.shape[1]
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model=CondDiffMLP(seq_dim, state_dim, hidden=a.hidden_dim, dropout=a.dropout).to(device)
    betas = cosine_beta_schedule(a.timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cum = torch.cumprod(alphas, dim=0)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    idxs = np.arange(seqs_norm.shape[0])
    for epoch in range(1, a.epochs+1):
        np.random.shuffle(idxs)
        losses=[]
        for i in tqdm(range(0, len(idxs), a.batch_size), desc=f"epoch {epoch}"):
            b = idxs[i:i+a.batch_size]
            if len(b)==0: continue
            clean = torch.from_numpy(seqs_norm[b]).to(device)
            s_b = torch.from_numpy(cond_states[b]).to(device)
            p_b = torch.from_numpy(phases[b]).to(device)
            t = torch.randint(0, a.timesteps, (clean.size(0),), device=device)
            alpha_bar = alphas_cum[t].unsqueeze(1)
            noise = torch.randn_like(clean)
            noisy = torch.sqrt(alpha_bar)*clean + torch.sqrt(1-alpha_bar)*noise
            pred = model(noisy, s_b, p_b, t, cond_drop_prob=a.cond_dropout)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            opt.step()
            losses.append(loss.item())
        print(f"epoch {epoch} loss={np.mean(losses):.6f}")
    out=Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "seq_dim": seq_dim,
        "state_dim": state_dim,
        "horizon": a.horizon,
        "timesteps": a.timesteps,
        "betas": betas.cpu(),
        "act_mean": act_mean,
        "act_std": act_std,
        "hidden_dim": a.hidden_dim
    }, out/"diffusion_policy_conditioned.pt")
    with open(out/"seq_stats.json","w") as f:
        json.dump({
            "samples": int(seqs.shape[0]),
            "horizon": a.horizon,
            "seq_dim": int(seq_dim),
            "state_dim": int(state_dim),
            "timesteps": a.timesteps
        }, f, indent=2)
    print("Saved", out/"diffusion_policy_conditioned.pt")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/raw/mj_pick_place_v5")
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--timesteps", type=int, default=200)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--cond_dropout", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_dir", type=str, default="models/ckpts_diffusion_cond_v2")
    a=ap.parse_args()
    main(a)
