import argparse, torch, json, numpy as np
from pathlib import Path
from envs.mj_pick_place_env import MjPickPlaceEnv, MjPickPlaceConfig
from models.bc_policy import SimpleBC
from models.critic import QNet
from utils.phase_labeling import label_phase

def load_bc(path, sdim, adim, device):
    m=SimpleBC(sdim,adim).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m

def load_q(path, sdim, adim, device):
    q=QNet(sdim, adim, twin=True).to(device)
    sd=torch.load(path, map_location=device)
    q.load_state_dict(sd, strict=False)
    q.eval()
    return q

def main(a):
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tmp_env=MjPickPlaceEnv(MjPickPlaceConfig(seed=a.seed))
    probe=tmp_env.reset()
    sdim=probe["state"].shape[0]; adim=4
    tmp_env.close()
    policy=load_bc(a.policy_ckpt, sdim, adim, device)
    qnet=load_q(a.qnet_ckpt, sdim, adim, device)
    out_dir=Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    env=MjPickPlaceEnv(MjPickPlaceConfig(seed=a.seed))
    for ep in range(a.episodes):
        obs=env.reset(); steps=0
        log=[]
        while True:
            st=obs["state"]; phase=label_phase(st)
            s_t=torch.from_numpy(st).float().unsqueeze(0).to(device)
            with torch.no_grad():
                act = policy(s_t).squeeze(0).cpu().numpy()
            cands=[act]
            for _ in range(a.adv_candidates-1):
                cands.append(np.clip(act + np.random.randn(*act.shape)*a.noise_scale, -1,1))
            s_b=torch.from_numpy(np.repeat(st[None,:], len(cands), axis=0)).float().to(device)
            a_b=torch.from_numpy(np.stack(cands)).float().to(device)
            p_b=torch.full((len(cands),), phase, dtype=torch.long, device=device)
            with torch.no_grad():
                q1,q2=qnet(s_b,a_b,p_b)
                qv=torch.min(q1,q2).cpu().numpy()
            best=int(np.argmax(qv))
            chosen=cands[best]
            adv = qv[best] - np.median(qv)
            obs,r,d,info=env.step(chosen)
            entry={"t":steps,"phase":phase,"chosen_idx":best,"adv":float(adv),"q_vals":qv.tolist(),"success":bool(info["success"]) if d else False}
            log.append(entry)
            steps+=1
            if d:
                fpath=out_dir/f"ep_{ep:03d}.json"
                with open(fpath,"w") as f: json.dump(log,f,indent=2)
                print("Saved log", fpath)
                break
    env.close()
    print("Completed logging.")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--policy_ckpt", type=str, default="models/ckpts_bc_base/bc_policy.pt")
    ap.add_argument("--qnet_ckpt", type=str, default="models/ckpts_iql_balanced_v4/qnet.pt")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--adv_candidates", type=int, default=8)
    ap.add_argument("--noise_scale", type=float, default=0.15)
    ap.add_argument("--out_dir", type=str, default="logs/rollout_advantage")
    a=ap.parse_args()
    main(a)