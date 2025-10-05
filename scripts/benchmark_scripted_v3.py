import argparse
from envs.mj_pick_place_env import MjPickPlaceEnv, MjPickPlaceConfig
from scripts.collect_mj_pick_place import (APPROACH, DESCEND, GRASP_WAIT, LIFT,
                                           MOVE_TO_TARGET, CENTER_FINE,
                                           fsm_policy)

def run(n, seed):
    cfg = MjPickPlaceConfig(seed=seed, debug=False)
    env = MjPickPlaceEnv(cfg)
    successes = 0
    for ep in range(n):
        obs = env.reset()
        fsm_state = APPROACH
        for t in range(cfg.max_steps):
            action, fsm_state = fsm_policy(obs, fsm_state, cfg)
            obs, r, d, info = env.step(action)
            if d:
                if info["success"]:
                    successes += 1
                break
    env.close()
    print(f"[Benchmark v3] Episodes={n} Successes={successes} SR={successes/n*100:.2f}%")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    run(args.episodes, args.seed)
