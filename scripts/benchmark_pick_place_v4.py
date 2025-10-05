import argparse
from envs.mj_pick_place_env import MjPickPlaceEnv, MjPickPlaceConfig
from scripts.collect_pick_place_v4 import (APPROACH, DESCEND, GRASP_LOCK, LIFT,
                                           MOVE_TARGET, FINE, policy_fsm)

def run(n, seed):
    cfg = MjPickPlaceConfig(seed=seed, lift_only=False, debug=False)
    env = MjPickPlaceEnv(cfg)
    successes=0
    for ep in range(n):
        obs=env.reset(); phase=APPROACH
        for t in range(cfg.max_steps):
            a, phase = policy_fsm(obs, phase, cfg)
            obs,r,d,info = env.step(a)
            if d:
                if info["success"]: successes+=1
                break
    env.close()
    print(f"[Benchmark v4] Episodes={n} Successes={successes} SR={successes/n*100:.2f}%")

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=10)
    a=ap.parse_args()
    run(a.episodes, a.seed)
