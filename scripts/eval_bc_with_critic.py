import argparse, torch, numpy as np
from envs.mj_pick_place_env import MjPickPlaceEnv, MjPickPlaceConfig
from models.bc_policy import SimpleBC
from models.critic import QNet, ValueNet
from utils.phase_labeling import label_phase

PHASE_ACTION_CANDS = {
    0: 4,  # APPROACH
    1: 6,  # DESCEND
    2: 8,  # GRASP_SETTLE
    3: 6,  # LIFT
    4: 10, # MOVE
    5: 12  # FINE
}

def load_bc(path, sdim, adim, device):
    m=SimpleBC(sdim, adim).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m

def load_qnet(path, sdim, adim, device):
    q=QNet(sdim, adim, twin=True).to(device)
    q.load_state_dict(torch.load(path, map_location=device))
    q.eval()
    return q

def choose_action(policy, qnet, state, phase, device, noise_scale, min_margin):
    with torch.no_grad():
        st = torch.from_numpy(state).float().unsqueeze(0).to(device)
        base = policy(st).squeeze(0).cpu().numpy()
    cand_num = PHASE_ACTION_CANDS.get(phase, 8)
    cands=[base]
    for _ in range(cand_num-1):
        cands.append(np.clip(base + np.random.randn(*base.shape)*noise_scale, -1,1))
    s_b = torch.from_numpy(np.repeat(state[None,:], len(cands), axis=0)).float().to(device)
    a_b = torch.from_numpy(np.stack(cands)).float().to(device)
    p_b = torch.full((len(cands),), phase, dtype=torch.long, device=device)
    with torch.no_grad():
        q1,q2 = qnet(s_b,a_b,p_b)
        q_vals = torch.min(q1,q2)
    q_np = q_vals.cpu().numpy()
    best = int(np.argmax(q_np))
    adv = q_np[best] - np.median(q_np)
    if adv < min_margin:
        return base
    return cands[best]

def run(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    env_tmp = MjPickPlaceEnv(MjPickPlaceConfig(seed=args.seed))
    probe = env_tmp.reset()
    sdim = probe["state"].shape[0]; adim=4
    env_tmp.close()
    policy = load_bc(args.policy_ckpt, sdim, adim, device)
    qnet = load_qnet(args.qnet_ckpt, sdim, adim, device)
    env = MjPickPlaceEnv(MjPickPlaceConfig(seed=args.seed))
    succ=0
    for ep in range(args.episodes):
        obs=env.reset(); steps=0
        while True:
            st = obs["state"]
            phase = label_phase(st)
            act = choose_action(policy, qnet, st, phase, device, args.noise_scale, args.min_adv_margin)
            obs,r,d,info = env.step(act)
            steps+=1
            if d:
                print(f"Episode {ep} success={info['success']} steps={steps}")
                succ += int(info["success"])
                break
    env.close()
    print(f"Adaptive Q-filtered SR: {succ/args.episodes*100:.2f}%")

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--policy_ckpt", type=str, default="models/ckpts_bc_base/bc_policy.pt")
    ap.add_argument("--qnet_ckpt", type=str, default="models/ckpts_iql_balanced_v2/qnet.pt")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--noise_scale", type=float, default=0.12)
    ap.add_argument("--min_adv_margin", type=float, default=0.02)
    args=ap.parse_args()
    run(args)
