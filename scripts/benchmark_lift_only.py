import argparse
import numpy as np
from envs.mj_pick_place_env import MjPickPlaceEnv, MjPickPlaceConfig

def simple_grasp_then_lift(episodes, seed):
    cfg = MjPickPlaceConfig(seed=seed, lift_only=True, debug=False)
    env = MjPickPlaceEnv(cfg)
    succ = 0
    for ep in range(episodes):
        obs = env.reset()
        # Phase 1: align above cube
        for _ in range(20):
            s = obs["state"]
            eef = s[0:3]; cube = s[4:7]
            target = np.array([cube[0], cube[1], 0.18])
            delta = target - eef
            act = np.array([
                np.clip(delta[0], -0.05, 0.05),
                np.clip(delta[1], -0.05, 0.05),
                np.clip(delta[2], -0.05, 0.05),
                0.0
            ], dtype=np.float32)
            obs, r, d, info = env.step(act)
            if d: break
        if info.get("success"): 
            succ += 1
            continue
        # Phase 2: descend + close
        for _ in range(25):
            s = obs["state"]
            eef = s[0:3]; cube = s[4:7]
            delta = np.array([cube[0]-eef[0], cube[1]-eef[1], (cube[2]+0.010)-eef[2]])
            act = np.array([
                np.clip(delta[0], -0.04, 0.04),
                np.clip(delta[1], -0.04, 0.04),
                np.clip(delta[2], -0.04, 0.04),
                1.0
            ], dtype=np.float32)
            obs, r, d, info = env.step(act)
            if d: break
        if info.get("success"):
            succ += 1
            continue
        # Phase 3: lift
        for _ in range(30):
            s = obs["state"]
            eef = s[0:3]; cube = s[4:7]
            delta = np.array([0,0,0.18 - eef[2]])
            act = np.array([
                0.0, 0.0,
                np.clip(delta[2], -0.05, 0.05),
                1.0
            ], dtype=np.float32)
            obs, r, d, info = env.step(act)
            if d: break
        if info.get("success"):
            succ += 1
    env.close()
    print(f"Lift-only success: {succ}/{episodes} = {succ/episodes*100:.2f}%")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    simple_grasp_then_lift(args.episodes, args.seed)
