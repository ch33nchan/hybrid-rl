import argparse, torch, numpy as np
from envs.mj_pick_place_env import MjPickPlaceEnv, MjPickPlaceConfig
from models.bc_policy import SimpleBC
from models.critic import QNet
from utils.phase_labeling import label_phase

PHASE_ACTION_CANDS={0:3,1:5,2:7,3:7,4:10,5:12}

def load_bc(path, sdim, adim, device):
    m=SimpleBC(sdim, adim).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m

def load_q(path, sdim, adim, device):
    q=QNet(sdim, adim, twin=True).to(device)
    sd=torch.load(path, map_location=device)
    q.load_state_dict(sd, strict=False)
    q.eval()
    return q

def choose_action(policy,qnet,state,phase,device,noise,min_adv,phase_stats):
    with torch.no_grad():
        st=torch.from_numpy(state).float().unsqueeze(0).to(device)
        base=policy(st).squeeze(0).cpu().numpy()
    cand_n=PHASE_ACTION_CANDS.get(phase,6)
    cands=[base]
    for i in range(cand_n-1):
        cands.append(np.clip(base + np.random.randn(*base.shape)*noise, -1,1))
    s_b=torch.from_numpy(np.repeat(state[None,:],len(cands),axis=0)).float().to(device)
    a_b=torch.from_numpy(np.stack(cands)).float().to(device)
    p_b=torch.full((len(cands),), phase, dtype=torch.long, device=device)
    with torch.no_grad():
        q1,q2 = qnet(s_b,a_b,p_b)
        qv = torch.min(q1,q2).cpu().numpy()
    stats = phase_stats.setdefault(phase, {"mean":0.0,"count":0})
    stats["mean"] = 0.95*stats["mean"] + 0.05*float(qv.mean())
    stats["count"] += 1
    norm_q = qv - stats["mean"]
    best=int(np.argmax(norm_q))
    adv = norm_q[best] - np.median(norm_q)
    if adv < min_adv:
        return base
    return cands[best]

def run(a):
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tmp_env=MjPickPlaceEnv(MjPickPlaceConfig(seed=a.seed))
    probe=tmp_env.reset()
    sdim=probe["state"].shape[0]; adim=4
    tmp_env.close()
    policy=load_bc(a.policy_ckpt, sdim, adim, device)
    qnet=load_q(a.qnet_ckpt, sdim, adim, device)
    env=MjPickPlaceEnv(MjPickPlaceConfig(seed=a.seed))
    successes=0; phase_stats={}
    for ep in range(a.episodes):
        obs=env.reset(); steps=0
        while True:
            st=obs["state"]; phase=label_phase(st)
            act=choose_action(policy,qnet,st,phase,device,a.noise_scale,a.min_adv_margin,phase_stats)
            obs,r,d,info=env.step(act); steps+=1
            if d:
                print(f"Episode {ep} success={info['success']} steps={steps}")
                successes+=int(info["success"])
                break
    env.close()
    print(f"Phase-Norm Q-filtered SR: {successes/a.episodes*100:.2f}%")
    print("Phase means:", {k:round(v["mean"],4) for k,v in phase_stats.items()})

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--policy_ckpt", type=str, default="models/ckpts_bc_base/bc_policy.pt")
    ap.add_argument("--qnet_ckpt", type=str, default="models/ckpts_iql_balanced_v4/qnet.pt")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--noise_scale", type=float, default=0.12)
    ap.add_argument("--min_adv_margin", type=float, default=0.02)
    a=ap.parse_args()
    run(a)