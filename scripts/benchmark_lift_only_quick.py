import argparse, numpy as np
from envs.mj_pick_place_env import MjPickPlaceEnv, MjPickPlaceConfig

def run(n, seed):
    cfg = MjPickPlaceConfig(seed=seed, lift_only=True, debug=False)
    env = MjPickPlaceEnv(cfg)
    successes=0
    for ep in range(n):
        obs = env.reset()
        phase=0
        for t in range(cfg.max_steps):
            s=obs["state"]; eef=s[0:3]; cube=s[4:7]
            if phase==0:
                target=np.array([cube[0], cube[1], 0.18])
                delta=target-eef
                if np.linalg.norm(delta[:2])<0.01: phase=1
            elif phase==1:
                target=np.array([cube[0], cube[1], cube[2]+0.010])
                delta=target-eef
                if abs(delta[2])<0.004: phase=2
            else:
                target=np.array([cube[0], cube[1], 0.20])
                delta=target-eef
            delta_xy=delta[:2]; nxy=np.linalg.norm(delta_xy)
            if nxy>0.05 and nxy>1e-8: delta_xy=delta_xy*(0.05/nxy)
            dz=np.clip(delta[2], -0.05, 0.05)
            g=1.0 if phase>=1 else 0.0
            act=np.array([delta_xy[0], delta_xy[1], dz, g], dtype=np.float32)
            obs,r,d,info=env.step(act)
            if d:
                if info["success"]: successes+=1
                break
    env.close()
    print(f"Lift-only benchmark: {successes}/{n} = {successes/n*100:.2f}%")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=77)
    a=ap.parse_args()
    run(a.episodes, a.seed)
