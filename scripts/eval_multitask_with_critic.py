import argparse, torch, numpy as np, os
from envs.mj_pick_place_env import MjPickPlaceEnv, MjPickPlaceConfig
from models.multitask_policy import MultiTaskPolicy
from models.critic import QNet
from utils.phase_labeling import label_phase

PHASE_ACTION_CANDS={0:1,1:4,2:6,3:6,4:10,5:14}

def load_multitask_bundle(path, state_dim, action_dim, device):
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt["ema"] if isinstance(ckpt, dict) and ckpt.get("ema") else ckpt["model"] if isinstance(ckpt, dict) and ckpt.get("model") else ckpt
    model = MultiTaskPolicy(state_dim, action_dim, num_phases=6)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model

def load_qnet(path, sdim, adim, device):
    q=QNet(sdim, adim, twin=True).to(device)
    sd=torch.load(path, map_location=device)
    q.load_state_dict(sd, strict=False)
    q.eval()
    return q

def near_success(state):
    eef = state[0:3]; cube=state[4:7]; tgt=state[7:9]
    dist = np.linalg.norm(cube[:2]-tgt)
    return (cube[2] > 0.10) and (dist < 0.065)

def choose_action(policy, qnet, state, phase, device, noise_scale, min_adv_margin):
    with torch.no_grad():
        s_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        base_a, phase_logits = policy(s_t)
        base_a = torch.tanh(base_a).squeeze(0).cpu().numpy()
    # early phases: no or minimal exploration
    cand_n = PHASE_ACTION_CANDS.get(phase, 6)
    if phase in (0,):  # APPROACH
        return base_a
    cands=[base_a]
    for k in range(cand_n-1):
        scale = noise_scale * (1.0 + 0.2*k*(phase>=4))
        cands.append(np.clip(base_a + np.random.randn(*base_a.shape)*scale, -1,1))
    s_b = torch.from_numpy(np.repeat(state[None,:], len(cands), axis=0)).float().to(device)
    a_b = torch.from_numpy(np.stack(cands)).float().to(device)
    p_b = torch.full((len(cands),), phase, dtype=torch.long, device=device)
    with torch.no_grad():
        q1,q2 = qnet(s_b,a_b,p_b)
        q_vals = torch.min(q1,q2)
    q_np = q_vals.cpu().numpy()
    best = int(np.argmax(q_np))
    adv = q_np[best] - np.median(q_np)
    if adv < min_adv_margin:
        return base_a
    return cands[best]

def run(args):
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    env_tmp = MjPickPlaceEnv(MjPickPlaceConfig(seed=args.seed))
    probe=env_tmp.reset()
    sdim=probe["state"].shape[0]; adim=4
    env_tmp.close()
    policy=load_multitask_bundle(args.policy_ckpt, sdim, adim, device)
    qnet=load_qnet(args.qnet_ckpt, sdim, adim, device)
    env = MjPickPlaceEnv(MjPickPlaceConfig(seed=args.seed))
    successes=0
    for ep in range(args.episodes):
        obs=env.reset(); steps=0
        while True:
            st=obs["state"]; phase=label_phase(st)
            if near_success(st) and phase >= 4:
                act = np.array([0,0,0,1],dtype=np.float32)
            else:
                act=choose_action(policy,qnet,st,phase,device,args.noise_scale,args.min_adv_margin)
            obs,r,d,info=env.step(act); steps+=1
            if d:
                print(f"Episode {ep} success={info['success']} steps={steps}")
                successes+=int(info["success"])
                break
    env.close()
    print(f"Multitask TwinQ Filtered SR: {successes/args.episodes*100:.2f}%")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--policy_ckpt", type=str, default="models/ckpts_multitask_balanced_v5/multitask_policy_best.pt")
    ap.add_argument("--qnet_ckpt", type=str, default="models/ckpts_iql_balanced_v4/qnet.pt")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--noise_scale", type=float, default=0.10)
    ap.add_argument("--min_adv_margin", type=float, default=0.015)
    args=ap.parse_args()
    run(args)