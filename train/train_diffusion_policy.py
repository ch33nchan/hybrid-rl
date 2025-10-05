import argparse, torch, numpy as np, math, json
from pathlib import Path
from tqdm import tqdm
from lerobot_dataset.pick_place_mj_builder import MjPickPlaceOfflineDataset
from models.diffusion.unet_policy import SmallUNet1D
import torch.nn.functional as F

def build_sequences(base, horizon, stride):
    seq_states=[]
    seq_actions=[]
    for i in range(len(base)):
        if i + horizon >= len(base): break
        ep_i, t_i = base.index[i]
        ep_j, t_j = base.index[i+horizon]
        if ep_i != ep_j:
            continue
        acts=[]
        for k in range(horizon):
            it_k = base[i+k]
            acts.append(it_k["action"])
        seq_actions.append(np.concatenate(acts, axis=0))
        seq_states.append(base[i]["obs_state"])
    return np.array(seq_states, dtype=np.float32), np.array(seq_actions, dtype=np.float32)

def cosine_beta_schedule(T, s=0.008):
    steps = T+1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x/T)+s)/(1+s) * math.pi*0.5)**2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:]/alphas_cumprod[:-1])
    return torch.clip(betas, 1e-5, 0.02)

def main(a):
    base = MjPickPlaceOfflineDataset(a.data_root, use_paraphrase=False)
    states, action_seqs = build_sequences(base, a.horizon, a.stride)
    print("Sequences:", action_seqs.shape)
    seq_dim = action_seqs.shape[1]
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model=SmallUNet1D(seq_dim=seq_dim, model_dim=a.model_dim).to(device)
    betas = cosine_beta_schedule(a.timesteps)
    alphas = 1.0 - betas
    alphas_cum = torch.cumprod(alphas, dim=0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    N = action_seqs.shape[0]
    idxs = np.arange(N)
    for epoch in range(1, a.epochs+1):
        np.random.shuffle(idxs)
        losses=[]
        for i in tqdm(range(0, N, a.batch_size), desc=f"epoch {epoch}"):
            batch_idx = idxs[i:i+a.batch_size]
            if len(batch_idx)==0: continue
            seq = torch.from_numpy(action_seqs[batch_idx]).to(device)
            t = torch.randint(0, a.timesteps, (seq.size(0),), device=device)
            alpha_cum = alphas_cum[t].unsqueeze(1)
            noise = torch.randn_like(seq)
            noisy = torch.sqrt(alpha_cum)*seq + torch.sqrt(1-alpha_cum)*noise
            pred = model(noisy, t)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"epoch {epoch} loss={np.mean(losses):.6f}")
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(),"seq_dim": seq_dim,"horizon": a.horizon,"timesteps": a.timesteps}, out_dir/"diffusion_policy.pt")
    with open(out_dir/"seq_stats.json","w") as f:
        json.dump({"horizon": a.horizon, "seq_dim": int(seq_dim)}, f, indent=2)
    print("Saved diffusion policy to", out_dir/"diffusion_policy.pt")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/raw/mj_pick_place_v5")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--timesteps", type=int, default=100)
    ap.add_argument("--model_dim", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out_dir", type=str, default="models/ckpts_diffusion")
    a=ap.parse_args()
    main(a)