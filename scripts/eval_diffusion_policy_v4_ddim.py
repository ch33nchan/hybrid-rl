import argparse, torch, numpy as np
from envs.mj_pick_place_env import MjPickPlaceEnv, MjPickPlaceConfig
from utils.phase_labeling import label_phase
import torch.nn as nn
from tqdm import tqdm

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim=dim
    def forward(self,t):
        device=t.device; half=self.dim//2
        freqs=torch.exp(-torch.arange(half,device=device)*(torch.log(torch.tensor(10000.))/(half-1)))
        args=t.float().unsqueeze(1)*freqs.unsqueeze(0); emb=torch.cat([torch.sin(args),torch.cos(args)],dim=-1)
        if self.dim%2: emb=torch.cat([emb,torch.zeros(emb.shape[0],1,device=device)],dim=-1)
        return emb

class ResBlock(nn.Module):
    def __init__(self, dim, time_dim, dropout=0.0):
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
    def forward(self, x, state, phase, t):
        temb = self.time_mlp(t)
        pemb = self.phase_emb(phase)
        h = torch.cat([x, state, pemb], dim=-1)
        h = self.in_proj(h)
        for block in self.blocks: h = block(h, temb)
        return self.out(h)

@torch.no_grad()
def ddim_sample(model, norm_state, phase_id, betas, act_mean, act_std, horizon, num_steps=50):
    device=norm_state.device; seq_dim=horizon*4
    x=torch.randn(1,seq_dim,device=device)
    alphas=1.0-betas; alphas_cum=torch.cumprod(alphas,dim=0)
    total_timesteps = betas.shape[0]
    times = torch.linspace(-1, total_timesteps-1, steps=num_steps+1).long()
    times = list(reversed(times.tolist()))
    time_pairs = list(zip(times[:-1], times[1:]))

    for time, time_next in time_pairs:
        ts = torch.full((1,), time, device=device, dtype=torch.long)
        ph = torch.full((1,), phase_id, device=device, dtype=torch.long)
        pred_noise = model(x, norm_state.unsqueeze(0), ph, ts)
        alpha = alphas_cum[time]
        alpha_next = alphas_cum[time_next] if time_next >=0 else torch.tensor(1.0, device=device)
        pred_x0 = (x - torch.sqrt(1-alpha)*pred_noise) / torch.sqrt(alpha)
        pred_x0 = torch.clamp(pred_x0, -1, 1)
        dir_xt = torch.sqrt(1. - alpha_next) * pred_noise
        x = torch.sqrt(alpha_next) * pred_x0 + dir_xt
    denorm=x.squeeze(0)*act_std+act_mean
    acts=denorm.view(horizon,4)
    return torch.clamp(acts,-1,1).cpu().numpy()

def run(a):
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt=torch.load(a.checkpoint,map_location=device)
    model=CondDiffusionNetV4(ckpt["seq_dim"],ckpt["state_dim"],model_dim=ckpt.get("hidden_dim",512),depth=ckpt.get("depth",2))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    betas=ckpt.get("betas").to(device)
    act_mean=torch.tensor(ckpt["act_mean"],device=device)
    act_std=torch.tensor(ckpt["act_std"],device=device)
    state_mean=torch.tensor(ckpt["state_mean"],device=device)
    state_std=torch.tensor(ckpt["state_std"],device=device)
    horizon=ckpt["horizon"]
    env=MjPickPlaceEnv(MjPickPlaceConfig(seed=a.seed))
    successes=0
    progress=tqdm(range(a.episodes), desc="eval episodes", unit="ep")
    for ep in progress:
        obs=env.reset(); steps=0; plan=[]
        while True:
            st=obs["state"]
            if len(plan)==0:
                phase_id=label_phase(st)
                state_vec=torch.from_numpy(st).float().to(device)
                norm_state=(state_vec-state_mean)/state_std
                seq=ddim_sample(model,norm_state,phase_id,betas,act_mean,act_std,horizon,num_steps=a.num_steps)
                plan=seq.tolist()
            act=np.array(plan.pop(0),dtype=np.float32)
            obs,r,d,info=env.step(act); steps+=1
            if steps%a.replan_freq==0:
                plan=[]
            if d:
                print(f"Episode {ep} success={info['success']} steps={steps}")
                successes+=int(info["success"])
                break
        progress.set_postfix({"success_rate": f"{successes/(ep+1)*100:.2f}%"})
        if (ep+1)%a.print_every==0:
            print(f"Interim SR: {successes/(ep+1)*100:.2f}%")
    env.close()
    print(f"DDIM v4 SR: {successes/a.episodes*100:.2f}%")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",type=str,default="models/ckpts_diffusion_cond_v4/diffusion_policy_v4.pt")
    ap.add_argument("--episodes",type=int,default=20)
    ap.add_argument("--seed",type=int,default=2025)
    ap.add_argument("--replan_freq",type=int,default=4)
    ap.add_argument("--num_steps",type=int,default=50)
    ap.add_argument("--print_every",type=int,default=5)
    a=ap.parse_args()
    run(a)
