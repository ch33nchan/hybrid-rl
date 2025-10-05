import argparse, numpy as np
from envs.mj_pick_place_env import MjPickPlaceEnv, MjPickPlaceConfig
from scripts.collect_mj_pick_place import determine_phase, policy

def run(n, seed):
    cfg = MjPickPlaceConfig(seed=seed, debug=False)
    env = MjPickPlaceEnv(cfg)
    successes = 0
    lengths = []
    for ep in range(n):
        obs = env.reset()
        for t in range(cfg.max_steps):
            ph = determine_phase(t, obs, cfg)
            act = policy(obs, ph, cfg)
            obs, r, d, info = env.step(act)
            if d:
                if info["success"]:
                    successes += 1
                lengths.append(t+1)
                break
    env.close()
    sr = successes / n
    print(f"Episodes: {n} Successes: {successes} SR={sr*100:.2f}% avg_len={sum(lengths)/len(lengths):.1f}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    run(args.episodes, args.seed)
