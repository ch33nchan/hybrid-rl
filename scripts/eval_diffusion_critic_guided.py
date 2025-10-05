import argparse, torch, numpy as np
from envs.mj_pick_place_env import MjPickPlaceEnv, MjPickPlaceConfig
from utils.phase_labeling import label_phase
import torch.nn as nn

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

class QNet(nn.Module):
    def __init__(self,s_dim,a_dim,h=256,phase_emb_dim=32): # Corrected phase_emb_dim
        super().__init__()
        self.embed_phase = nn.Embedding(6, phase_emb_dim)
        self.q1=nn.Sequential(nn.Linear(s_dim+a_dim+phase_emb_dim,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,1))
        self.q2=nn.Sequential(nn.Linear(s_dim+a_dim+phase_emb_dim,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,1))
    def forward(self,s,a,p):
        p_emb = self.embed_phase(p)
        x = torch.cat([s,a,p_emb],-1)
        return self.q1(x), self.q2(x)

@torch.no_grad()
def ddim_ensemble_generate(model, norm_state, phase_id, betas, act_mean, act_std, horizon, num_ensemble, num_steps=50):
    device=norm_state.device; seq_dim=horizon*4
    norm_state_batch = norm_state.unsqueeze(0).repeat(num_ensemble, 1)
    phase_id_batch = torch.full((num_ensemble,), phase_id, device=device, dtype=torch.long)
    x = torch.randn(num_ensemble, seq_dim, device=device)
    alphas=1.0-betas; alphas_cum=torch.cumprod(alphas,dim=0)
    total_timesteps = betas.shape[0]
    times = torch.linspace(-1, total_timesteps-1, steps=num_steps+1).long()
    time_pairs = list(zip(times[:-1].tolist(), times[1:].tolist()))
    for time, time_next in time_pairs:
        ts = torch.full((num_ensemble,), time, device=device, dtype=torch.long)
        pred_noise = model(x, norm_state_batch, phase_id_batch, ts)
        alpha = alphas_cum[time]
        alpha_next = alphas_cum[time_next] if time_next >=0 else torch.tensor(1.0, device=device)
        pred_x0 = (x - torch.sqrt(1-alpha)*pred_noise) / torch.sqrt(alpha)
        dir_xt = torch.sqrt(1. - alpha_next) * pred_noise
        x = torch.sqrt(alpha_next) * pred_x0 + dir_xt
    denorm = x * act_std + act_mean
    return torch.clamp(denorm.view(num_ensemble, horizon, 4), -1, 1)

def run(a):
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    diff_ckpt=torch.load(a.diff_checkpoint,map_location=device)
    diff_model=CondDiffusionNetV4(diff_ckpt["seq_dim"],diff_ckpt["state_dim"],model_dim=diff_ckpt.get("hidden_dim",512),depth=diff_ckpt.get("depth",4))
    diff_model.load_state_dict(diff_ckpt["model"]); diff_model.to(device).eval()

    critic_sdim, adim = 9, 4 # Corrected state dimension for critic
    critic = QNet(critic_sdim, adim).to(device)
    critic.load_state_dict(torch.load(a.qnet_ckpt, map_location=device))
    critic.eval()

    betas=diff_ckpt.get("betas").to(device)
    act_mean=torch.tensor(diff_ckpt["act_mean"],device=device)
    act_std=torch.tensor(diff_ckpt["act_std"],device=device)
    state_mean=torch.tensor(diff_ckpt["state_mean"],device=device)
    state_std=torch.tensor(diff_ckpt["state_std"],device=device)
    horizon=diff_ckpt["horizon"]
    env=MjPickPlaceEnv(MjPickPlaceConfig(seed=a.seed))
    successes=0
    for ep in range(a.episodes):
        obs=env.reset(); steps=0; plan=[]
        while True:
            st=obs["state"]
            if len(plan)==0:
                phase_id=label_phase(st)
                state_vec=torch.from_numpy(st).float().to(device)
                norm_state=(state_vec[:9]-state_mean)/state_std
                
                candidate_plans = ddim_ensemble_generate(diff_model,norm_state,phase_id,betas,act_mean,act_std,horizon,a.num_ensemble,num_steps=a.num_steps)
                
                with torch.no_grad():
                    first_actions = candidate_plans[:, 0, :]
                    state_batch = norm_state.unsqueeze(0).repeat(a.num_ensemble, 1) # Use normalized state for critic
                    phase_batch = torch.full((a.num_ensemble,), phase_id, device=device, dtype=torch.long)
                    q1, q2 = critic(state_batch, first_actions, phase_batch)
                    scores = torch.min(q1, q2).squeeze()
                
                best_plan_idx = torch.argmax(scores).item()
                plan = candidate_plans[best_plan_idx].cpu().numpy().tolist()

            act=np.array(plan.pop(0),dtype=np.float32)
            obs,r,d,info=env.step(act); steps+=1
            if steps % a.replan_freq == 0: plan=[]
            if d:
                print(f"Episode {ep} success={info['success']} steps={steps}")
                successes+=int(info["success"])
                break
        if (ep+1)%a.print_every==0: print(f"Interim SR: {successes/(ep+1)*100:.2f}%")
    env.close()
    print(f"Critic-Guided Diffusion SR: {successes/a.episodes*100:.2f}%")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--diff_checkpoint",type=str,default="models/ckpts_diffusion_cond_v4/diffusion_policy_v4.pt")
    ap.add_argument("--qnet_ckpt",type=str,default="models/ckpts_iql_balanced_v4/qnet.pt")
    ap.add_argument("--episodes",type=int,default=20)
    ap.add_argument("--seed",type=int,default=2025)
    ap.add_argument("--replan_freq",type=int,default=4)
    ap.add_argument("--num_steps",type=int,default=50)
    ap.add_argument("--num_ensemble",type=int,default=16)
    ap.add_argument("--print_every",type=int,default=5)
    a=ap.parse_args()
    run(a)
