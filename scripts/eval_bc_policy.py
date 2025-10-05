import argparse
import numpy as np
import torch
from pathlib import Path
from envs.mj_pick_place_env import MjPickPlaceEnv, MjPickPlaceConfig
from models.policy_loader import load_bc_policy

def run_episode(env, model, device, max_steps=None, deterministic=True, record=False):
    obs = env.reset()
    frames = []
    if record:
        frames.append(obs["rgb"])
    steps = 0
    success = False
    if max_steps is None:
        max_steps = env.cfg.max_steps
    for t in range(max_steps):
        state = torch.from_numpy(obs["state"]).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action = model(state)[0].cpu().numpy()
        action = np.tanh(action)
        obs, reward, done, info = env.step(action)
        if record:
            frames.append(obs["rgb"])
        steps += 1
        if done:
            success = info["success"]
            break
    return {"success": success, "steps": steps, "frames": frames if record else None}

def main(args):
    tmp_env = MjPickPlaceEnv(MjPickPlaceConfig(seed=args.seed))
    probe = tmp_env.reset()
    state_dim = probe["state"].shape[0]
    action_dim = 4
    tmp_env.close()

    model, device = load_bc_policy(args.checkpoint, state_dim, action_dim)

    results = []
    env = MjPickPlaceEnv(MjPickPlaceConfig(seed=args.seed))
    for ep in range(args.episodes):
        ep_result = run_episode(env, model, device, record=args.record)
        results.append(ep_result)
        print(f"Episode {ep}: success={ep_result['success']} steps={ep_result['steps']}")
    env.close()

    success_rate = sum(r["success"] for r in results) / len(results)
    avg_steps = sum(r["steps"] for r in results) / len(results)
    print(f"Success Rate: {success_rate*100:.2f}% | Avg Steps: {avg_steps:.1f}")

    if args.record and results[0]["frames"]:
        out_root = Path(args.out_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        import numpy as np
        np.save(out_root / "episode0_frames.npy", np.stack(results[0]["frames"]))
        print("Saved frames ->", out_root / "episode0_frames.npy")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="models/ckpts/bc_policy.pt")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--out_dir", type=str, default="eval_outputs")
    args = ap.parse_args()
    main(args)
