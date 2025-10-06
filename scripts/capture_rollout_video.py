"""
Capture video frames from policy rollouts for report visualization.
Creates side-by-side comparisons and trajectory visualizations.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from tqdm import tqdm

from envs.mj_pick_place_env import MjPickPlaceEnv, MjPickPlaceConfig
from models.multitask_policy import MultiTaskPolicy
from models.critic import QNet
from utils.phase_labeling import label_phase, PHASES

PHASE_NAMES = ["APPROACH", "DESCEND", "GRASP", "LIFT", "MOVE", "FINE"]
PHASE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


def load_policy(ckpt_path, state_dim, action_dim, device):
    """Load multitask policy."""
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("ema") or ckpt.get("model") or ckpt
    policy = MultiTaskPolicy(state_dim, action_dim, num_phases=6)
    policy.load_state_dict(state_dict, strict=False)
    policy.to(device).eval()
    return policy


def load_critic(ckpt_path, state_dim, action_dim, device):
    """Load IQL critic."""
    qnet = QNet(state_dim, action_dim, twin=True).to(device)
    qnet.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    qnet.eval()
    return qnet


def run_episode_with_policy(env, policy, qnet, device, use_critic=True, noise_scale=0.1):
    """Run a single episode and collect frames + metadata."""
    obs = env.reset()
    frames = []
    states = []
    actions_taken = []
    phases = []
    rewards = []
    
    step = 0
    done = False
    
    while not done and step < 160:
        state = obs["state"]
        phase = label_phase(state)
        
        # Get action
        with torch.no_grad():
            s_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
            pred_action, _ = policy(s_t)
            action = torch.tanh(pred_action).squeeze(0).cpu().numpy()
        
        # Critic-guided selection if enabled
        if use_critic and qnet is not None and phase > 0:
            num_candidates = min(6, 2 + phase)
            candidates = [action]
            for _ in range(num_candidates - 1):
                candidates.append(np.clip(action + np.random.randn(4) * noise_scale, -1, 1))
            
            s_b = torch.from_numpy(np.repeat(state[None, :], len(candidates), axis=0)).float().to(device)
            a_b = torch.from_numpy(np.stack(candidates)).float().to(device)
            p_b = torch.full((len(candidates),), phase, dtype=torch.long, device=device)
            
            with torch.no_grad():
                q1, q2 = qnet(s_b, a_b, p_b)
                q_vals = torch.min(q1, q2)
            
            best_idx = int(torch.argmax(q_vals).cpu())
            action = candidates[best_idx]
        
        # Step environment
        obs, reward, done, info = env.step(action)
        
        # Record
        frames.append(obs["rgb"])
        states.append(state)
        actions_taken.append(action)
        phases.append(phase)
        rewards.append(reward)
        
        step += 1
    
    return {
        "frames": frames,
        "states": np.array(states),
        "actions": np.array(actions_taken),
        "phases": np.array(phases),
        "rewards": np.array(rewards),
        "success": info.get("success", False),
        "steps": step
    }


def create_rollout_visualization(rollout_data, out_path, title="Policy Rollout"):
    """Create comprehensive visualization of a single rollout."""
    frames = rollout_data["frames"]
    states = rollout_data["states"]
    phases = rollout_data["phases"]
    actions = rollout_data["actions"]
    
    # Select key frames (every 10 steps)
    key_indices = list(range(0, len(frames), 10))
    if key_indices[-1] != len(frames) - 1:
        key_indices.append(len(frames) - 1)
    
    num_frames = len(key_indices)
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(4, num_frames, hspace=0.3, wspace=0.2)
    
    # Top row: RGB frames
    for i, idx in enumerate(key_indices):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(frames[idx])
        phase_name = PHASE_NAMES[phases[idx]]
        ax.set_title(f't={idx}\n{phase_name}', fontsize=9, fontweight='bold')
        ax.axis('off')
    
    # Second row: State evolution
    ax_state = fig.add_subplot(gs[1, :])
    timesteps = np.arange(len(states))
    
    ax_state.plot(timesteps, states[:, 2], label='EEF Z', linewidth=2, color='blue')
    ax_state.plot(timesteps, states[:, 6], label='Cube Z', linewidth=2, color='red')
    ax_state.axhline(y=0.105, color='green', linestyle='--', alpha=0.5, label='Success Height')
    
    # Color background by phase
    phase_changes = np.where(np.diff(phases) != 0)[0] + 1
    phase_regions = [0] + list(phase_changes) + [len(phases)]
    for i in range(len(phase_regions) - 1):
        start, end = phase_regions[i], phase_regions[i + 1]
        phase_id = phases[start]
        ax_state.axvspan(start, end, alpha=0.15, color=PHASE_COLORS[phase_id])
    
    ax_state.set_xlabel('Timestep', fontweight='bold')
    ax_state.set_ylabel('Z Position (m)', fontweight='bold')
    ax_state.set_title('Height Evolution', fontweight='bold')
    ax_state.legend(loc='upper left')
    ax_state.grid(alpha=0.3)
    
    # Third row: Distance to target and gripper
    ax_dist = fig.add_subplot(gs[2, :])
    dist_to_target = np.linalg.norm(states[:, 4:6] - states[:, 7:9], axis=1)
    
    ax_dist.plot(timesteps, dist_to_target, label='XY Distance to Target', linewidth=2, color='purple')
    ax_dist.axhline(y=0.055, color='green', linestyle='--', alpha=0.5, label='Success Radius')
    ax_dist.plot(timesteps, states[:, 3] * 0.1, label='Gripper (scaled)', linewidth=2, color='orange')
    
    ax_dist.set_xlabel('Timestep', fontweight='bold')
    ax_dist.set_ylabel('Distance (m)', fontweight='bold')
    ax_dist.set_title('Distance to Target & Gripper State', fontweight='bold')
    ax_dist.legend(loc='upper right')
    ax_dist.grid(alpha=0.3)
    
    # Fourth row: Actions
    ax_actions = fig.add_subplot(gs[3, :])
    action_labels = ['X', 'Y', 'Z', 'Gripper']
    for i, label in enumerate(action_labels):
        ax_actions.plot(timesteps, actions[:, i], label=label, linewidth=1.5, alpha=0.8)
    
    ax_actions.set_xlabel('Timestep', fontweight='bold')
    ax_actions.set_ylabel('Action Value', fontweight='bold')
    ax_actions.set_title('Action Commands', fontweight='bold')
    ax_actions.legend(loc='upper right', ncol=4)
    ax_actions.grid(alpha=0.3)
    ax_actions.set_ylim([-1.1, 1.1])
    
    # Overall title
    success_str = "✓ SUCCESS" if rollout_data["success"] else "✗ FAILED"
    fig.suptitle(f'{title} - {success_str} ({rollout_data["steps"]} steps)', 
                 fontsize=16, fontweight='bold')
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_comparison_grid(rollouts, out_path, titles):
    """Create side-by-side comparison of multiple rollouts."""
    num_rollouts = len(rollouts)
    
    fig, axes = plt.subplots(num_rollouts, 8, figsize=(20, 3 * num_rollouts))
    if num_rollouts == 1:
        axes = axes[np.newaxis, :]
    
    for row, (rollout, title) in enumerate(zip(rollouts, titles)):
        frames = rollout["frames"]
        phases = rollout["phases"]
        
        # Select 8 evenly spaced frames
        indices = np.linspace(0, len(frames) - 1, 8, dtype=int)
        
        for col, idx in enumerate(indices):
            ax = axes[row, col]
            ax.imshow(frames[idx])
            
            if row == 0:
                ax.set_title(f't={idx}', fontsize=10, fontweight='bold')
            
            if col == 0:
                success_str = "✓" if rollout["success"] else "✗"
                ax.set_ylabel(f'{title}\n{success_str}', fontsize=10, fontweight='bold')
            
            # Add phase label
            phase_name = PHASE_NAMES[phases[idx]]
            ax.text(0.5, 0.95, phase_name, transform=ax.transAxes,
                   ha='center', va='top', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor=PHASE_COLORS[phases[idx]], alpha=0.7))
            
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def main(args):
    """Main execution."""
    print("="*60)
    print("CAPTURING POLICY ROLLOUT VISUALIZATIONS")
    print("="*60)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize environment
    env = MjPickPlaceEnv(MjPickPlaceConfig(seed=args.seed, render_width=256, render_height=256))
    probe = env.reset()
    state_dim = probe["state"].shape[0]
    action_dim = 4
    
    # Load models
    print("\nLoading models...")
    policy = load_policy(args.policy_ckpt, state_dim, action_dim, device)
    qnet = load_critic(args.qnet_ckpt, state_dim, action_dim, device) if args.use_critic else None
    print("✓ Models loaded")
    
    # Collect rollouts
    print(f"\nCollecting {args.num_episodes} rollout(s)...")
    rollouts = []
    
    for ep in tqdm(range(args.num_episodes), desc="Running episodes"):
        rollout = run_episode_with_policy(env, policy, qnet, device, 
                                         use_critic=args.use_critic,
                                         noise_scale=args.noise_scale)
        rollouts.append(rollout)
        
        # Create individual visualization
        title = f"Episode {ep+1}"
        if args.use_critic:
            title += " (BC+Critic)"
        else:
            title += " (BC Only)"
        
        out_path = out_dir / f"rollout_ep{ep+1}.png"
        create_rollout_visualization(rollout, out_path, title=title)
        print(f"  ✓ Episode {ep+1}: {'SUCCESS' if rollout['success'] else 'FAILED'} ({rollout['steps']} steps) -> {out_path.name}")
    
    env.close()
    
    # Create comparison grid if multiple episodes
    if args.num_episodes > 1:
        print("\nCreating comparison grid...")
        titles = [f"Ep{i+1}" for i in range(len(rollouts))]
        grid_path = out_dir / "rollout_comparison.png"
        create_comparison_grid(rollouts, grid_path, titles)
        print(f"✓ Saved: {grid_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("ROLLOUT STATISTICS")
    print("="*60)
    successes = sum(r["success"] for r in rollouts)
    print(f"Success Rate: {successes}/{args.num_episodes} ({successes/args.num_episodes*100:.1f}%)")
    print(f"Average Steps: {np.mean([r['steps'] for r in rollouts]):.1f}")
    print(f"Output Directory: {out_dir.absolute()}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture policy rollout visualizations")
    parser.add_argument("--policy_ckpt", type=str, 
                       default="models/ckpts_multitask_balanced_v4/multitask_policy.pt")
    parser.add_argument("--qnet_ckpt", type=str,
                       default="models/ckpts_iql_balanced_v4/qnet.pt")
    parser.add_argument("--output_dir", type=str, default="figures/rollouts")
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_critic", action="store_true", default=True)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    
    args = parser.parse_args()
    main(args)
