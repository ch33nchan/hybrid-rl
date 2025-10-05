import argparse, torch, numpy as np
from envs.mj_pick_place_env import MjPickPlaceEnv, MjPickPlaceConfig
from models.policy_loader import load_bc_policy

def main(a):
    env = MjPickPlaceEnv(MjPickPlaceConfig(seed=a.seed))
    probe = env.reset()
    state_dim = probe["state"].shape[0]
    action_dim = 4
    env.close()
    model, device = load_bc_policy(a.checkpoint, state_dim, action_dim)
    env = MjPickPlaceEnv(MjPickPlaceConfig(seed=a.seed))
    successes = 0
    for ep in range(1, a.episodes+1):
        obs = env.reset()
        for t in range(env.cfg.max_steps):
            s = torch.from_numpy(obs["state"]).float().unsqueeze(0).to(device)
            with torch.no_grad():
                act = np.tanh(model(s)[0].cpu().numpy())
            obs, r, d, info = env.step(act)
            if d:
                if info["success"]:
                    successes += 1
                break
        print(f"Episode {ep} success={info['success']} running_SR={successes/ep:.2f}")
        if successes >= a.stop_after_successes > 0:
            break
    env.close()
    print(f"Final Success Rate: {successes/ep:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="models/ckpts/bc_policy.pt")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--stop_after_successes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=999)
    a = ap.parse_args()
    main(a)
