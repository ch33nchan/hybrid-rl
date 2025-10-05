import argparse, torch, numpy as np
from envs.mj_pick_place_env import MjPickPlaceEnv, MjPickPlaceConfig
import torch.nn.functional as F
from utils.phase_labeling import label_phase

class SinusoidalTimeEmbedding(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim=dim
    def forward(self, t):
        device=t.device
        half=self.dim//2
        freqs = torch.exp(-torch.arange(half, device=device)*(torch.log(torch.tensor(10000.0))/(half-1)))
        args = t.float().unsqueeze(1)*freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim%2==1:
            emb = torch.cat([emb, torch.zeros(emb.shape[0],1,device=device)], dim=-1)
        return emb

class CondMLP(torch.nn.Module):
    def __init__(self, seq_dim, state_dim, phase_emb=16, time_dim=128, hidden=512):
        super().__init__()
        self.time = torch.nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            torch.nn.Linear(time_dim, hidden),
            torch.nn.SiLU()
        )
        self.phase_emb = torch.nn.Embedding(6, phase_emb)
        self.in_proj = torch.nn.Linear(seq_dim + state_dim + phase_emb, hidden)
        self.fc1 = torch.nn.Linear(hidden, hidden)
        self.fc2 = torch.nn.Linear(hidden, hidden)
        self.out = torch.nn.Linear(hidden, seq_dim)
        self.act = torch.nn.SiLU()
    def forward(self, noisy_seq, cond_state, phase_ids, t):
        temb = self.time(t)
        pemb = self.phase_emb(phase_ids)
        x = torch.cat([noisy_seq, cond_state, pemb], dim=-1)
        h = self.in_proj(x) + temb
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        return self.out(h)

@torch.no_grad()
def sample_action_sequence(model, state_vec, phase_id, horizon, timesteps, act_mean, act_std, device, guidance_scale=1.0):
    seq_dim = horizon * 4
    x = torch.randn(1, seq_dim, device=device)
    state_vec = state_vec.unsqueeze(0)
    phase_t = torch.tensor([phase_id], device=device, dtype=torch.long)
    for t in reversed(range(timesteps)):
        ts = torch.full((1,), t, device=device, dtype=torch.long)
        eps = model(x, state_vec, phase_t, ts)
        x = x - 0.1 * eps * guidance_scale
    denorm = x.squeeze(0) * act_std + act_mean
    acts = denorm.view(horizon, 4)
    return torch.clamp(acts, -1, 1).cpu().numpy()

def run(a):
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(a.checkpoint, map_location=device)
    horizon = ckpt["horizon"]
    seq_dim = ckpt["seq_dim"]
    state_dim = ckpt["state_dim"]
    model = CondMLP(seq_dim, state_dim, hidden=ckpt.get("hidden_dim",512))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    act_mean = torch.tensor(ckpt["act_mean"], device=device)
    act_std  = torch.tensor(ckpt["act_std"], device=device)
    env = MjPickPlaceEnv(MjPickPlaceConfig(seed=a.seed))
    successes=0
    for ep in range(a.episodes):
        obs=env.reset()
        steps=0
        action_buffer=[]
        while True:
            st = obs["state"]
            if steps % horizon == 0:
                phase_id = label_phase(st)
                state_t = torch.from_numpy(st).float().to(device)
                seq = sample_action_sequence(
                    model, state_t, phase_id, horizon,
                    ckpt["timesteps"], act_mean, act_std,
                    device, guidance_scale=a.guidance_scale
                )
                action_buffer = seq.tolist()
            act = np.array(action_buffer[steps % horizon], dtype=np.float32)
            obs,r,d,info=env.step(act)
            steps+=1
            if d:
                print(f"Episode {ep} success={info['success']} steps={steps}")
                successes += int(info["success"])
                break
        if (ep+1) % a.print_every ==0:
            print(f"Interim SR: {successes/(ep+1)*100:.2f}%")
    env.close()
    print(f"Conditioned Diffusion SR: {successes/a.episodes*100:.2f}%")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="models/ckpts_diffusion_cond/diffusion_policy_conditioned.pt")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--guidance_scale", type=float, default=1.0)
    ap.add_argument("--print_every", type=int, default=5)
    a=ap.parse_args()
    run(a)
