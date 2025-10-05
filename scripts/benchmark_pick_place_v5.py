import argparse, numpy as np
from envs.mj_pick_place_env import MjPickPlaceEnv, MjPickPlaceConfig
from scripts.collect_pick_place_v5 import (APPROACH,DESCEND,GRASP_SETTLE,LIFT,MOVE,FINE,
                                           choose_phase,phase_target,action_from_setpoint)

def run(n, seed):
    cfg = MjPickPlaceConfig(seed=seed, lift_only=False, debug=False)
    env = MjPickPlaceEnv(cfg)
    succ=0
    for ep in range(n):
        obs=env.reset(); phase=APPROACH
        for t in range(cfg.max_steps):
            s=obs["state"]; eef=s[0:3]; gr=s[3]; cube=s[4:7]; tgt=s[7:9]
            attached = (gr>0.5 and np.linalg.norm(eef-cube)<0.05) or np.linalg.norm(eef-cube)<0.02
            phase = choose_phase(eef,cube,tgt,attached,phase)
            setpt = phase_target(phase,eef,cube,tgt)
            a = action_from_setpoint(eef,setpt,phase)
            obs,r,d,info=env.step(a)
            if d:
                if info["success"]: succ+=1
                break
    env.close()
    print(f"[Benchmark v5] Episodes={n} Success={succ} SR={succ/n*100:.2f}%")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    a=ap.parse_args()
    run(a.episodes,a.seed)
