import argparse, json, cv2
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import trange
from envs.mj_pick_place_env import MjPickPlaceEnv, MjPickPlaceConfig

# FSM States
APPROACH = 0
DESCEND  = 1
GRASP_WAIT = 2
LIFT = 3
MOVE_TO_TARGET = 4
CENTER_FINE = 5

def horizontal_err(eef, target_xy):
    return np.linalg.norm(eef[:2] - target_xy)

def fsm_policy(obs, state, env_cfg):
    s = obs["state"]
    eef = s[0:3]
    gripper = s[3]
    cube = s[4:7]
    tgt = s[7:9]
    # Derived
    horiz_to_cube = horizontal_err(eef, cube[:2])
    attached_inferred = (gripper > 0.5 and np.linalg.norm(eef - cube) < 0.04) or (np.linalg.norm(eef - cube) < 0.02)

    # Transition logic
    if state == APPROACH:
        if horiz_to_cube < 0.01 and eef[2] > cube[2] + 0.015:
            state = DESCEND
    elif state == DESCEND:
        if eef[2] <= cube[2] + 0.013:
            state = GRASP_WAIT
    elif state == GRASP_WAIT:
        if attached_inferred:
            state = LIFT
    elif state == LIFT:
        if eef[2] >= 0.17:
            state = MOVE_TO_TARGET
    elif state == MOVE_TO_TARGET:
        if horizontal_err(eef, tgt) < 0.02:
            state = CENTER_FINE
    elif state == CENTER_FINE:
        # Stay until env success triggers (no explicit release required)
        pass

    # Action synthesis per state
    g = 1.0 if state >= GRASP_WAIT else 0.0
    target = eef.copy()

    if state == APPROACH:
        target = np.array([cube[0], cube[1], 0.18])
        g = 0.0
    elif state == DESCEND:
        target = np.array([cube[0], cube[1], cube[2] + 0.010])
        g = 1.0  # start closing slightly early
    elif state == GRASP_WAIT:
        target = np.array([cube[0], cube[1], cube[2] + 0.010])
        g = 1.0
    elif state == LIFT:
        target = np.array([cube[0], cube[1], 0.18])
        g = 1.0
    elif state == MOVE_TO_TARGET:
        target = np.array([tgt[0], tgt[1], 0.18])
        g = 1.0
    elif state == CENTER_FINE:
        target = np.array([tgt[0], tgt[1], 0.18])
        g = 1.0

    delta = target - eef
    # Limit horizontal and vertical separately
    max_h = 0.05
    max_v = 0.05
    delta_xy = delta[:2]
    norm_xy = np.linalg.norm(delta_xy)
    if norm_xy > max_h and norm_xy > 1e-8:
        delta_xy = delta_xy * (max_h / norm_xy)
    dz = np.clip(delta[2], -max_v, max_v)
    act = np.array([delta_xy[0], delta_xy[1], dz, g], dtype=np.float32)
    return act, state

def collect(episodes, output, seed):
    cfg = MjPickPlaceConfig(seed=seed, debug=False)
    env = MjPickPlaceEnv(cfg)
    out_dir = Path(output); out_dir.mkdir(parents=True, exist_ok=True)
    successes = 0

    for ep in trange(episodes, desc="Collect v3"):
        obs = env.reset()
        fsm_state = APPROACH
        frames = [obs["rgb"]]
        states = [obs["state"]]
        actions = []
        dones = []
        success = False

        for t in range(cfg.max_steps):
            action, fsm_state = fsm_policy(obs, fsm_state, cfg)
            obs, r, d, info = env.step(action)
            actions.append(action)
            frames.append(obs["rgb"])
            states.append(obs["state"])
            dones.append(d)
            if d:
                success = info["success"]
                break

        if success:
            successes += 1

        ep_dir = out_dir / f"episode_{ep:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            ep_dir / "trajectory.npz",
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
            "instruction": "pick and place the red cube onto the green target (v3 FSM no-release)"
        }
        with open(ep_dir / "meta.json","w") as f:
            json.dump(meta, f, indent=2)
        cv2.imwrite(str(ep_dir/"preview.png"), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))

    env.close()
    print(f"Scripted v3 success: {successes}/{episodes} = {successes/episodes*100:.2f}%")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--output", type=str, default="data/raw/mj_pick_place_v3")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    collect(args.episodes, args.output, args.seed)
