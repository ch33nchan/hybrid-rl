import argparse, json, cv2
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import trange
from envs.mj_pick_place_env import MjPickPlaceEnv, MjPickPlaceConfig

APPROACH=0
DESCEND=1
GRASP_SETTLE=2
LIFT=3
MOVE=4
FINE=5

def horiz_err(eef, cube): return np.linalg.norm(eef[:2]-cube[:2])
def dist_xy(a,b): return np.linalg.norm(a-b)

def choose_phase(eef, cube, tgt, attached, phase):
    if phase==APPROACH and horiz_err(eef,cube)<0.012: return DESCEND
    if phase==DESCEND and abs(eef[2] - (cube[2]+0.010))<0.004: return GRASP_SETTLE
    if phase==GRASP_SETTLE and attached: return LIFT
    if phase==LIFT and eef[2]>0.185: return MOVE
    if phase==MOVE and dist_xy(eef[:2], tgt)<0.03: return FINE
    return phase

def phase_target(phase,eef,cube,tgt):
    hover=0.19
    close_z=cube[2]+0.010
    if phase==APPROACH: return np.array([cube[0], cube[1], hover])
    if phase==DESCEND:  return np.array([cube[0], cube[1], close_z])
    if phase==GRASP_SETTLE: return np.array([cube[0], cube[1], close_z])
    if phase==LIFT: return np.array([cube[0], cube[1], hover])
    if phase==MOVE: return np.array([tgt[0], tgt[1], hover])
    if phase==FINE: return np.array([tgt[0], tgt[1], hover])
    return eef

def action_from_setpoint(eef, setpoint, phase):
    err = setpoint - eef
    scale_xy = 18.0
    scale_z  = 18.0
    a = np.array([
        np.clip(err[0]*scale_xy, -1,1),
        np.clip(err[1]*scale_xy, -1,1),
        np.clip(err[2]*scale_z,  -1,1),
        1.0 if phase>=DESCEND else 0.0
    ], dtype=np.float32)
    return a

def collect(episodes, output, seed):
    cfg = MjPickPlaceConfig(seed=seed, lift_only=False, debug=False)
    env = MjPickPlaceEnv(cfg)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    successes=0
    for ep in trange(episodes, desc="Collect v5 BOOST"):
        obs = env.reset()
        phase=APPROACH
        frames=[obs["rgb"]]; states=[obs["state"]]; actions=[]; dones=[]
        success=False
        for t in range(cfg.max_steps):
            s=obs["state"]
            eef=s[0:3]; gripper=s[3]; cube=s[4:7]; tgt=s[7:9]
            attached = (gripper>0.5 and np.linalg.norm(eef-cube)<0.05) or np.linalg.norm(eef-cube)<0.02
            phase = choose_phase(eef,cube,tgt,attached,phase)
            setpt = phase_target(phase,eef,cube,tgt)
            a = action_from_setpoint(eef,setpt,phase)
            obs,r,d,info = env.step(a)
            actions.append(a); frames.append(obs["rgb"]); states.append(obs["state"]); dones.append(d)
            if d:
                success=info["success"]; break
        if success: successes+=1
        ep_dir = out / f"episode_{ep:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ep_dir/"trajectory.npz",
            obs_rgb=np.stack(frames).astype(np.uint8),
            obs_state=np.stack(states).astype(np.float32),
            actions=np.stack(actions).astype(np.float32),
            dones=np.array(dones,dtype=bool),
            success=np.array([success],dtype=bool)
        )
        meta = {
            "episode": ep,
            "success": bool(success),
            "steps": len(actions),
            "seed": seed+ep,
            "timestamp": datetime.utcnow().isoformat()+"Z",
            "instruction": "pick and place the red cube onto the green target (v5 servo)"
        }
        with open(ep_dir/"meta.json","w") as f: json.dump(meta,f,indent=2)
        cv2.imwrite(str(ep_dir/"preview.png"), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
    env.close()
    print(f"Scripted v5 success {successes}/{episodes} = {successes/episodes*100:.2f}%")

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--output", type=str, default="data/raw/mj_pick_place_v5_boost")
    ap.add_argument("--seed", type=int, default=42)
    args=ap.parse_args()
    collect(args.episodes,args.output,args.seed)
