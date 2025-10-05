import argparse, json, cv2, numpy as np, os, tempfile
from pathlib import Path
from datetime import datetime
from tqdm import trange
from envs.mj_pick_place_env import MjPickPlaceEnv, MjPickPlaceConfig

def atomic_json_write(path: Path, data: dict):
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".meta_tmp_", suffix=".json")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.remove(tmp_name)
        except OSError:
            pass
        raise

def unstable(data):
    # Basic numeric sanity check
    if not np.isfinite(data).all():
        return True
    if np.linalg.norm(data) > 1e6:
        return True
    return False

def descend_grasp_sequence(env, steps=60):
    obs = env.reset()
    if unstable(obs["state"]):
        return None
    frames=[obs["rgb"]]; states=[obs["state"]]; actions=[]; dones=[]; success=False
    for t in range(steps):
        s = obs["state"]
        eef=s[0:3]; cube=s[4:7]
        horiz = cube[:2] - eef[:2]
        a_xy = np.clip(horiz*45.0, -1, 1)
        z_target = cube[2] + (0.015 if t<30 else 0.010)
        dz = np.clip( (z_target - eef[2]) * 45.0, -1, 1)
        g = 0.0 if t < 12 else 1.0
        act = np.array([a_xy[0], a_xy[1], dz, g], dtype=np.float32)
        obs,r,d,info = env.step(act)
        if unstable(obs["state"]):
            return None
        actions.append(act); frames.append(obs["rgb"]); states.append(obs["state"]); dones.append(d)
        if d:
            success = bool(info["success"])
            break
    return frames, states, actions, dones, success

def fine_align_sequence(env, steps=70):
    obs = env.reset()
    if unstable(obs["state"]):
        return None
    frames=[obs["rgb"]]; states=[obs["state"]]; actions=[]; dones=[]; success=False
    for t in range(steps):
        s = obs["state"]; eef=s[0:3]; gripper=s[3]; cube=s[4:7]; tgt=s[7:9]
        attached = (gripper>0.5 and np.linalg.norm(eef-cube)<0.05) or (np.linalg.norm(eef-cube)<0.02)
        if not attached:
            # phased approach
            if t < 10:
                target = np.array([cube[0], cube[1], 0.19])
                g=0.0
            elif t < 25:
                target = np.array([cube[0], cube[1], cube[2]+0.012])
                g=1.0 if t>18 else 0.0
            else:
                target = np.array([cube[0], cube[1], cube[2]+0.010])
                g=1.0
        else:
            base = np.array([tgt[0], tgt[1], 0.19])
            jitter = 0.004*np.array([np.sin(t*0.25), np.cos(t*0.31), 0])
            target = base + jitter
            g=1.0
        err = target - eef
        act = np.array([
            np.clip(err[0]*30,-1,1),
            np.clip(err[1]*30,-1,1),
            np.clip(err[2]*30,-1,1),
            g
        ], dtype=np.float32)
        obs,r,d,info = env.step(act)
        if unstable(obs["state"]):
            return None
        actions.append(act); frames.append(obs["rgb"]); states.append(obs["state"]); dones.append(d)
        if d:
            success = bool(info["success"])
            break
    return frames, states, actions, dones, success

def collect(descend_n, fine_n, out_dir, seed, max_retries=15):
    root = Path(out_dir); root.mkdir(parents=True, exist_ok=True)
    env_d = MjPickPlaceEnv(MjPickPlaceConfig(seed=seed))
    env_f = MjPickPlaceEnv(MjPickPlaceConfig(seed=seed+100))
    ep_count=0
    # DESCEND/GRASP
    for i in trange(descend_n, desc="Booster DESCEND_GRASP"):
        tries=0
        while tries < max_retries:
            result = descend_grasp_sequence(env_d)
            if result is not None:
                frames, states, actions, dones, success = result
                ep_dir = root / f"episode_desc_{i:03d}"
                ep_dir.mkdir(exist_ok=True)
                np.savez_compressed(ep_dir/"trajectory.npz",
                    obs_rgb=np.stack(frames).astype(np.uint8),
                    obs_state=np.stack(states).astype(np.float32),
                    actions=np.stack(actions).astype(np.float32),
                    dones=np.array(dones,dtype=bool),
                    success=np.array([bool(success)],dtype=bool)
                )
                meta = {
                    "type":"descend_grasp_boost",
                    "success": bool(success),
                    "steps": len(actions),
                    "timestamp": datetime.utcnow().isoformat()+"Z"
                }
                atomic_json_write(ep_dir/"meta.json", meta)
                ep_count+=1
                break
            tries+=1
    # MOVE/FINE
    for j in trange(fine_n, desc="Booster MOVE_FINE"):
        tries=0
        while tries < max_retries:
            result = fine_align_sequence(env_f)
            if result is not None:
                frames, states, actions, dones, success = result
                ep_dir = root / f"episode_fine_{j:03d}"
                ep_dir.mkdir(exist_ok=True)
                np.savez_compressed(ep_dir/"trajectory.npz",
                    obs_rgb=np.stack(frames).astype(np.uint8),
                    obs_state=np.stack(states).astype(np.float32),
                    actions=np.stack(actions).astype(np.float32),
                    dones=np.array(dones,dtype=bool),
                    success=np.array([bool(success)],dtype=bool)
                )
                meta = {
                    "type":"move_fine_boost",
                    "success": bool(success),
                    "steps": len(actions),
                    "timestamp": datetime.utcnow().isoformat()+"Z"
                }
                atomic_json_write(ep_dir/"meta.json", meta)
                ep_count+=1
                break
            tries+=1
    env_d.close(); env_f.close()
    print("Booster episodes created:", ep_count)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--descend_n", type=int, default=40)
    ap.add_argument("--fine_n", type=int, default=40)
    ap.add_argument("--out_dir", type=str, default="data/raw/mj_pick_place_phase_boost")
    ap.add_argument("--seed", type=int, default=555)
    args = ap.parse_args()
    collect(args.descend_n, args.fine_n, args.out_dir, args.seed)
