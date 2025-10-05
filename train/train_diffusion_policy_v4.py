import argparse, torch, numpy as np, math, json, random
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from lerobot_dataset.pick_place_mj_builder import MjPickPlaceOfflineDataset
from utils.phase_labeling import label_phase

def cosine_beta_schedule(T,s=0.008):
    steps=T+1; x=torch.linspace(0,T,steps); alphas_cumprod=torch.cos(((x/T)+s)/(1+s)*math.pi*0.5)**2
    alphas_cumprod=alphas_cumprod/alphas_cumprod[0]; betas=1-(alphas_cumprod[1:]/alphas_cumprod[:-1])
    return torch.clip(betas, 1e-5, 0.02)

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim=dim
    def forward(self,t):
        device=t.device; half=self.dim//2
        freqs=torch.exp(-torch.arange(half,device=device)*(torch.log(torch.tensor(10000.))/(half-1)))
        args=t.float().unsqueeze(1)*freqs.unsqueeze(0); emb=torch.cat([torch.sin(args),torch.cos(args)],dim=-1)
        if self.dim%2: emb=torch.cat([emb,torch.zeros(emb.shape[0],1,device=device)],dim=-1)
        return emb

class ResBlock(nn.Module):
    def __init__(self, dim, time_dim, dropout=0.1):
        super().__init__()
        self.in_block = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.LayerNorm(dim))
        self.time_proj = nn.Linear(time_dim, dim)
        self.out_block = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Dropout(dropout))
    def forward(self, x, t_emb):
        h = self.in_block(x)
        h = h + self.time_proj(t_emb)
        return self.out_block(h) + x

class CondDiffusionNetV4(nn.Module):
    def __init__(self, seq_dim, state_dim, phase_emb=16, model_dim=512, time_dim=128, depth=4):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalTimeEmbedding(time_dim), nn.Linear(time_dim, model_dim), nn.SiLU())
        self.phase_emb = nn.Embedding(6, phase_emb)
        self.in_proj = nn.Linear(seq_dim + state_dim + phase_emb, model_dim)
        self.blocks = nn.ModuleList([ResBlock(model_dim, model_dim) for _ in range(depth)])
        self.out = nn.Sequential(nn.Linear(model_dim, model_dim//2), nn.SiLU(), nn.Linear(model_dim//2, seq_dim))
    def forward(self, x, state, phase, t, drop_prob=0.1):
        if self.training and drop_prob > 0:
            mask = (torch.rand_like(state[:,0]) < drop_prob).float().unsqueeze(1)
            state = state * (1-mask)
        temb = self.time_mlp(t)
        pemb = self.phase_emb(phase)
        h = torch.cat([x, state, pemb], dim=-1)
        h = self.in_proj(h)
        for block in self.blocks: h = block(h, temb)
        return self.out(h)

def build_sequences(base, horizon):
    seqs, conds, phases = [], [], []
    for i in range(len(base) - horizon):
        ep_i, t_i = base.index[i]
        ep_j, t_j = base.index[i+horizon-1]
        if ep_i != ep_j: continue
        s0 = base[i]["obs_state"]
        acts = [base[i+k]["action"] for k in range(horizon)]
        seqs.append(np.concatenate(acts)); conds.append(s0); phases.append(label_phase(s0))
    return np.array(seqs,dtype=np.float32), np.array(conds,dtype=np.float32), np.array(phases,dtype=np.int64)

def normalize(data):
    mean=data.mean(0,keepdims=True); std=data.std(0,keepdims=True)+1e-6
    return (data-mean)/std, mean.squeeze(0), std.squeeze(0)

def main(a):
    base=MjPickPlaceOfflineDataset(a.data_root,use_paraphrase=False)
    seqs,conds,phases=build_sequences(base,a.horizon)
    seqs_norm,act_mean,act_std=normalize(seqs)
    states_norm,state_mean,state_std=normalize(conds)
    seq_dim=seqs_norm.shape[1]; state_dim=states_norm.shape[1]
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model=CondDiffusionNetV4(seq_dim,state_dim,model_dim=a.hidden_dim,depth=a.depth).to(device)
    betas=cosine_beta_schedule(a.timesteps).to(device)
    alphas=1.0-betas; alphas_cum=torch.cumprod(alphas,dim=0)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    idxs=np.arange(len(seqs_norm))
    for epoch in range(1,a.epochs+1):
        np.random.shuffle(idxs); losses=[]
        for i in tqdm(range(0,len(idxs),a.batch_size),desc=f"epoch {epoch}"):
            b=idxs[i:i+a.batch_size]
            clean=torch.from_numpy(seqs_norm[b]).to(device)
            state_b=torch.from_numpy(states_norm[b]).to(device)
            phase_b=torch.from_numpy(phases[b]).to(device)
            t=torch.randint(0,a.timesteps,(len(b),),device=device)
            alpha_bar=alphas_cum[t].unsqueeze(1)
            noise=torch.randn_like(clean)
            noisy=torch.sqrt(alpha_bar)*clean+torch.sqrt(1-alpha_bar)*noise
            pred=model(noisy,state_b,phase_b,t,drop_prob=a.cond_dropout)
            loss=F.mse_loss(pred,noise)
            opt.zero_grad(); loss.backward()
            opt.step()
            losses.append(loss.item())
        scheduler.step()
        print(f"epoch {epoch} loss={np.mean(losses):.6f} lr={scheduler.get_last_lr()[0]:.6f}")
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    torch.save({
        "model":model.state_dict(),"seq_dim":seq_dim,"state_dim":state_dim,"horizon":a.horizon,
        "timesteps":a.timesteps,"betas":betas.cpu(),"act_mean":act_mean,"act_std":act_std,
        "state_mean":state_mean,"state_std":state_std,"hidden_dim":a.hidden_dim, "depth":a.depth
    }, out/"diffusion_policy_v4.pt")
    print("Saved", out/"diffusion_policy_v4.pt")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    a=ap.parse_args()
    a.data_root="data/raw/mj_pick_place_v5"; a.horizon=8; a.timesteps=100; a.hidden_dim=512
    a.depth=4; a.batch_size=128; a.lr=3e-4; a.epochs=15; a.cond_dropout=0.1
    a.out_dir="models/ckpts_diffusion_cond_v4"
    main(a)
