import argparse, json, cv2
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import trange
from envs.mj_pick_place_env import MjPickPlaceEnv, MjPickPlaceConfig

def collect(episodes, output, seed):
    cfg = MjPickPlaceConfig(seed=seed, lift_only=True, debug=False)
    env = MjPickPlaceEnv(cfg)
    out_dir = Path(output); out_dir.mkdir(parents=True, exist_ok=True)
    successes = 0
    for ep in trange(episodes, desc="Collect lift"):
        obs = env.reset()
        frames = [obs["rgb"]]; states=[obs["state"]]; actions=[]; dones=[]
        success=False
        phase=0
        for t in range(cfg.max_steps):
            s = obs["state"]
            eef = s[0:3]; cube = s[4:7]
            # Phase 0: move XY above cube at hover
            if phase == 0:
                target = np.array([cube[0], cube[1], 0.18])
                delta = target - eef
                if np.linalg.norm(delta[:2]) < 0.01:
                    phase = 1
            # Phase 1: descend & close
            if phase == 1:
                target = np.array([cube[0], cube[1], cube[2] + 0.010])
                delta = target - eef
                if abs(delta[2]) < 0.004:
                    phase = 2
            # Phase 2: lift up
            if phase == 2:
                target = np.array([cube[0], cube[1], 0.20])
                delta = target - eef
            # Action synth
            delta_xy = delta[:2]
            nxy = np.linalg.norm(delta_xy)
            if nxy > 0.05 and nxy > 1e-8:
                delta_xy = delta_xy * (0.05 / nxy)
            dz = np.clip(delta[2], -0.05, 0.05)
            g = 1.0 if phase >= 1 else 0.0
            act = np.array([delta_xy[0], delta_xy[1], dz, g], dtype=np.float32)
            obs,r,d,info = env.step(act)
            actions.append(act)
            frames.append(obs["rgb"])
            states.append(obs["state"])
            dones.append(d)
            if d:
                success = info["success"]
                break
        if success: successes += 1
        ep_dir = out_dir / f"episode_{ep:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ep_dir / "trajectory.npz",
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
            "seed": seed + ep,
            "timestamp": datetime.utcnow().isoformat()+"Z",
            "instruction": "lift the red cube (lift-only bootstrap)"
        }
        with open(ep_dir / "meta.json","w") as f: json.dump(meta,f,indent=2)
        cv2.imwrite(str(ep_dir/"preview.png"), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
    env.close()
    print(f"Lift-only scripted success: {successes}/{episodes} = {successes/episodes*100:.2f}%")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--output", type=str, default="data/raw/mj_lift_only_v0")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    collect(args.episodes, args.output, args.seed)
