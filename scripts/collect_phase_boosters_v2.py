import argparse, json, cv2, numpy as np, os, tempfile
from pathlib import Path
from datetime import datetime
from tqdm import trange
from envs.mj_pick_place_env import MjPickPlaceEnv, MjPickPlaceConfig

def atomic_json(path: Path, data: dict):
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    with os.fdopen(tmp_fd, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_name, path)

def grasp_settle_clip(env, steps=40):
    obs = env.reset()
    frames=[obs["rgb"]]; states=[obs["state"]]; actions=[]; dones=[]
    success=False
    for t in range(steps):
        s = obs["state"]
        eef=s[0:3]; cube=s[4:7]
        target = np.array([cube[0], cube[1], cube[2]+0.014])
        horiz = target[:2] - eef[:2]
        dz = (target[2]-eef[2])
        a = np.array([
            np.clip(horiz[0]*50,-1,1),
            np.clip(horiz[1]*50,-1,1),
            np.clip(dz*50,-1,1),
            1.0 if t>8 else 0.0
        ], dtype=np.float32)
        if t>15:
            a[:3] *= 0.1
        obs,r,d,info = env.step(a)
        frames.append(obs["rgb"]); states.append(obs["state"]); actions.append(a); dones.append(d)
        if d:
            success=info["success"]
            break
    return frames, states, actions, dones, success

def move_fine_clip(env, steps=60):
    obs = env.reset()
    frames=[obs["rgb"]]; states=[obs["state"]]; actions=[]; dones=[]
    success=False
    phase=0
    for t in range(steps):
        s=obs["state"]; eef=s[0:3]; cube=s[4:7]; tgt=s[7:9]; grip=s[3]
        dist_cube = np.linalg.norm(eef - cube)
        attached = (grip>0.5 and dist_cube<0.05) or dist_cube<0.02
        if not attached:
            if phase==0:
                target = np.array([cube[0], cube[1], 0.19]); g=0.0
                if np.linalg.norm(eef[:2]-cube[:2])<0.012:
                    phase=1
            elif phase==1:
                target = np.array([cube[0], cube[1], cube[2]+0.012]); g=1.0 if t>12 else 0.0
        else:
            radial = 0.03 * max(0, 1 - t/steps)
            angle = 0.25 * t
            offset = np.array([radial*np.cos(angle), radial*np.sin(angle)])
            base = np.array([tgt[0], tgt[1]]) + offset
            target = np.array([base[0], base[1], 0.19]); g=1.0
        err = target - eef
        a = np.array([
            np.clip(err[0]*30,-1,1),
            np.clip(err[1]*30,-1,1),
            np.clip(err[2]*30,-1,1),
            g
        ], dtype=np.float32)
        obs,r,d,info = env.step(a)
        frames.append(obs["rgb"]); states.append(obs["state"]); actions.append(a); dones.append(d)
        if d:
            success=info["success"]
            break
    return frames, states, actions, dones, success

def collect(gs_n, mv_n, out_dir, seed):
    root=Path(out_dir); root.mkdir(parents=True, exist_ok=True)
    env_g = MjPickPlaceEnv(MjPickPlaceConfig(seed=seed))
    env_m = MjPickPlaceEnv(MjPickPlaceConfig(seed=seed+100))
    ep=0
    for i in trange(gs_n, desc="GRASP_SETTLE Boost"):
        f,s,a,d,success = grasp_settle_clip(env_g)
        ep_dir = root/f"gs_{i:03d}"
        ep_dir.mkdir(exist_ok=True)
        np.savez_compressed(ep_dir/"trajectory.npz", obs_rgb=np.stack(f), obs_state=np.stack(s),
                            actions=np.stack(a), dones=np.array(d), success=np.array([success],dtype=bool))
        atomic_json(ep_dir/"meta.json",{
            "type":"grasp_settle_boost","success":bool(success),
            "steps":len(a),"timestamp":datetime.utcnow().isoformat()+"Z"})
        ep+=1
    for j in trange(mv_n, desc="MOVE_FINE Boost"):
        f,s,a,d,success = move_fine_clip(env_m)
        ep_dir = root/f"mv_{j:03d}"
        ep_dir.mkdir(exist_ok=True)
        np.savez_compressed(ep_dir/"trajectory.npz", obs_rgb=np.stack(f), obs_state=np.stack(s),
                            actions=np.stack(a), dones=np.array(d), success=np.array([success],dtype=bool))
        atomic_json(ep_dir/"meta.json",{
            "type":"move_fine_boost","success":bool(success),
            "steps":len(a),"timestamp":datetime.utcnow().isoformat()+"Z"})
        ep+=1
    env_g.close(); env_m.close()
    print("Total booster clips:", ep)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--grasp_settle", type=int, default=120)
    ap.add_argument("--move_fine", type=int, default=160)
    ap.add_argument("--out_dir", type=str, default="data/raw/mj_pick_place_boost_v2")
    ap.add_argument("--seed", type=int, default=777)
    a=ap.parse_args()
    collect(a.grasp_settle, a.move_fine, a.out_dir, a.seed)