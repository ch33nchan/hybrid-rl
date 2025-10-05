import argparse, json, cv2
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import trange
from envs.mj_pick_place_env import MjPickPlaceEnv, MjPickPlaceConfig

# FSM states
APPROACH = 0
DESCEND = 1
GRASP_LOCK = 2
LIFT = 3
MOVE_TARGET = 4
FINE = 5

def horiz_err(a,b):
    return np.linalg.norm(a[:2]-b[:2])

def policy_fsm(obs, phase, cfg):
    s = obs["state"]
    eef = s[0:3]
    gripper = s[3]
    cube = s[4:7]
    tgt = s[7:9]
    attached = (gripper > 0.5 and np.linalg.norm(eef - cube) < 0.05) or np.linalg.norm(eef - cube) < 0.02
    hover_z = 0.19
    close_z = cube[2] + 0.010

    # Transitions
    if phase == APPROACH and horiz_err(eef, cube) < 0.01:
        phase = DESCEND
    if phase == DESCEND and abs(eef[2] - close_z) < 0.004:
        phase = GRASP_LOCK
    if phase == GRASP_LOCK and attached:
        phase = LIFT
    if phase == LIFT and eef[2] >= hover_z - 0.01:
        phase = MOVE_TARGET
    if phase == MOVE_TARGET and horiz_err(eef, tgt) < 0.025:
        phase = FINE

    if phase == APPROACH:
        target = np.array([cube[0], cube[1], hover_z])
        g = 0.0
    elif phase == DESCEND:
        target = np.array([cube[0], cube[1], close_z])
        g = 1.0  # start closing
    elif phase == GRASP_LOCK:
        target = np.array([cube[0], cube[1], close_z])
        g = 1.0
    elif phase == LIFT:
        target = np.array([cube[0], cube[1], hover_z])
        g = 1.0
    elif phase == MOVE_TARGET:
        target = np.array([tgt[0], tgt[1], hover_z])
        g = 1.0
    else:  # FINE
        target = np.array([tgt[0], tgt[1], hover_z])
        g = 1.0

    delta = target - eef
    max_h = 0.045
    max_v = 0.045
    dxy = delta[:2]
    nxy = np.linalg.norm(dxy)
    if nxy > max_h and nxy > 1e-8:
        dxy = dxy * (max_h / nxy)
    dz = np.clip(delta[2], -max_v, max_v)
    action = np.array([dxy[0], dxy[1], dz, g], dtype=np.float32)
    return action, phase

def collect(episodes, output, seed):
    cfg = MjPickPlaceConfig(seed=seed, lift_only=False, debug=False)
    env = MjPickPlaceEnv(cfg)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    successes = 0
    for ep in trange(episodes, desc="Collect v4"):
        obs = env.reset()
        phase = APPROACH
        frames=[obs["rgb"]]; states=[obs["state"]]; actions=[]; dones=[]
        success=False
        for t in range(cfg.max_steps):
            a, phase = policy_fsm(obs, phase, cfg)
            obs, r, d, info = env.step(a)
            actions.append(a); frames.append(obs["rgb"]); states.append(obs["state"]); dones.append(d)
            if d:
                success = info["success"]
                break
        if success: successes += 1
        ep_dir = out / f"episode_{ep:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ep_dir/"trajectory.npz",
            obs_rgb=np.stack(frames).astype(np.uint8),
            obs_state=np.stack(states).astype(np.float32),
            actions=np.stack(actions).astype(np.float32),
            dones=np.array(dones, dtype=bool),
            success=np.array([success], dtype=bool)
        )
        meta = {
            "episode": ep,
            "success": bool(success),
            "steps": len(actions),
            "seed": seed+ep,
            "timestamp": datetime.utcnow().isoformat()+"Z",
            "instruction": "pick and place the red cube onto the green target (v4)"
        }
        with open(ep_dir/"meta.json","w") as f: json.dump(meta,f,indent=2)
        cv2.imwrite(str(ep_dir/"preview.png"), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
    env.close()
    print(f"Scripted v4 success {successes}/{episodes} = {successes/episodes*100:.2f}%")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--output", type=str, default="data/raw/mj_pick_place_v4")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    collect(args.episodes, args.output, args.seed)
