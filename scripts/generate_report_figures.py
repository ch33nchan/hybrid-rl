"""
Generate comprehensive figures and visualizations for the experiment report.
Creates:
1. Phase distribution bar chart
2. Trajectory visualization (state evolution)
3. Environment screenshots from different phases
4. Action distribution plots
5. Success rate comparison chart
6. Training curves (if logs available)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from collections import Counter
import json
from tqdm import tqdm

from lerobot_dataset.pick_place_mj_builder import MjPickPlaceOfflineDataset
from utils.phase_labeling import PHASES, label_phase
from envs.mj_pick_place_env import MjPickPlaceEnv, MjPickPlaceConfig

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
PHASE_NAMES = ["APPROACH", "DESCEND", "GRASP", "LIFT", "MOVE", "FINE"]


def create_output_dir(out_dir):
    """Create output directory for figures."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def plot_phase_distribution(data_root, out_dir):
    """Plot phase distribution in the dataset."""
    print("\n[1/7] Generating phase distribution plot...")
    ds = MjPickPlaceOfflineDataset(data_root, use_paraphrase=False, verbose=False)
    
    phase_counts = Counter()
    for i in tqdm(range(len(ds)), desc="Counting phases"):
        phase_counts[ds[i]["phase_id"]] += 1
    
    # Prepare data
    phases = sorted(phase_counts.keys())
    counts = [phase_counts[p] for p in phases]
    labels = [PHASE_NAMES[p] for p in phases]
    total = sum(counts)
    percentages = [c/total*100 for c in counts]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, counts, color=COLORS[:len(labels)], alpha=0.8, edgecolor='black')
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Task Phase', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('Phase Distribution in Offline Dataset', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add total count
    ax.text(0.98, 0.98, f'Total Samples: {total:,}', 
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    save_path = out_dir / 'phase_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_trajectory_evolution(data_root, out_dir, num_episodes=5):
    """Plot state evolution for sample trajectories."""
    print("\n[2/7] Generating trajectory evolution plots...")
    ds = MjPickPlaceOfflineDataset(data_root, use_paraphrase=False, verbose=False)
    
    # Load a few complete episodes
    episodes_data = []
    ep_dirs = sorted(Path(data_root).glob("episode_*"))[:num_episodes]
    
    for ep_dir in tqdm(ep_dirs, desc="Loading episodes"):
        traj_file = ep_dir / "trajectory.npz"
        if traj_file.exists():
            data = np.load(traj_file)
            episodes_data.append(data)
    
    if not episodes_data:
        print("⚠ No episodes found, skipping trajectory plot")
        return
    
    # Create multi-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for ep_data in episodes_data:
        states = ep_data["obs_state"]
        T = len(states)
        timesteps = np.arange(T)
        
        # Extract components
        eef_z = states[:, 2]
        cube_z = states[:, 6]
        gripper = states[:, 3]
        dist_to_target = np.linalg.norm(states[:, 4:6] - states[:, 7:9], axis=1)
        
        # Plot EEF and Cube Z-height
        axes[0, 0].plot(timesteps, eef_z, alpha=0.6, linewidth=1.5)
        axes[0, 1].plot(timesteps, cube_z, alpha=0.6, linewidth=1.5)
        axes[1, 0].plot(timesteps, gripper, alpha=0.6, linewidth=1.5)
        axes[1, 1].plot(timesteps, dist_to_target, alpha=0.6, linewidth=1.5)
    
    axes[0, 0].set_title('End-Effector Z-Height', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylabel('Z Position (m)')
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].set_title('Cube Z-Height', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Timestep')
    axes[0, 1].set_ylabel('Z Position (m)')
    axes[0, 1].axhline(y=0.105, color='red', linestyle='--', alpha=0.5, label='Success Threshold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    axes[1, 0].set_title('Gripper State', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].set_ylabel('Gripper (0=Open, 1=Closed)')
    axes[1, 0].set_ylim([-0.1, 1.1])
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].set_title('Distance to Target', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Timestep')
    axes[1, 1].set_ylabel('XY Distance (m)')
    axes[1, 1].axhline(y=0.055, color='red', linestyle='--', alpha=0.5, label='Success Radius')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    fig.suptitle('Trajectory State Evolution (Sample Episodes)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = out_dir / 'trajectory_evolution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_action_distributions(data_root, out_dir, max_samples=10000):
    """Plot action distribution statistics."""
    print("\n[3/7] Generating action distribution plots...")
    ds = MjPickPlaceOfflineDataset(data_root, use_paraphrase=False, verbose=False)
    
    actions = []
    phases = []
    for i in tqdm(range(min(len(ds), max_samples)), desc="Loading actions"):
        sample = ds[i]
        actions.append(sample["action"])
        phases.append(sample["phase_id"])
    
    actions = np.array(actions)
    phases = np.array(phases)
    
    # Create figure with subplots for each action dimension
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    action_labels = ['X-axis', 'Y-axis', 'Z-axis', 'Gripper']
    
    for idx, (ax, label) in enumerate(zip(axes.flat, action_labels)):
        # Overall distribution
        ax.hist(actions[:, idx], bins=50, alpha=0.6, color=COLORS[0], 
                edgecolor='black', label='All Phases')
        
        ax.set_xlabel(f'{label} Action Value', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'Action Distribution: {label}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
    
    fig.suptitle('Action Distribution Across Dataset', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = out_dir / 'action_distributions.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_phase_action_heatmap(data_root, out_dir, max_samples=10000):
    """Plot heatmap of action magnitudes per phase."""
    print("\n[4/7] Generating phase-action heatmap...")
    ds = MjPickPlaceOfflineDataset(data_root, use_paraphrase=False, verbose=False)
    
    # Collect action magnitudes per phase
    phase_actions = {p: [] for p in range(6)}
    
    for i in tqdm(range(min(len(ds), max_samples)), desc="Analyzing phase actions"):
        sample = ds[i]
        phase_id = sample["phase_id"]
        action = sample["action"]
        phase_actions[phase_id].append(np.abs(action))
    
    # Compute mean absolute action per phase
    mean_actions = np.zeros((6, 4))
    for p in range(6):
        if phase_actions[p]:
            mean_actions[p] = np.mean(phase_actions[p], axis=0)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mean_actions, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(6))
    ax.set_xticklabels(['X-axis', 'Y-axis', 'Z-axis', 'Gripper'], fontsize=11)
    ax.set_yticklabels(PHASE_NAMES, fontsize=11)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Absolute Action', fontsize=11, fontweight='bold')
    
    # Add text annotations
    for i in range(6):
        for j in range(4):
            text = ax.text(j, i, f'{mean_actions[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    ax.set_title('Mean Action Magnitude by Phase', fontsize=14, fontweight='bold')
    ax.set_xlabel('Action Dimension', fontsize=12, fontweight='bold')
    ax.set_ylabel('Task Phase', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_path = out_dir / 'phase_action_heatmap.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def capture_environment_screenshots(out_dir):
    """Capture screenshots from the environment at different phases."""
    print("\n[5/7] Capturing environment screenshots...")
    
    env = MjPickPlaceEnv(MjPickPlaceConfig(seed=42, render_width=256, render_height=256))
    
    # Define scripted actions to reach different phases
    phase_scripts = {
        "initial": [],
        "approach": [(0, 0, 0, 0)] * 10,
        "descend": [(0, 0, 0, 0)] * 10 + [(0, 0, -1, 0)] * 15,
        "grasp": [(0, 0, 0, 0)] * 10 + [(0, 0, -1, 0)] * 15 + [(0, 0, 0, 1)] * 5,
        "lift": [(0, 0, 0, 0)] * 10 + [(0, 0, -1, 0)] * 15 + [(0, 0, 0, 1)] * 5 + [(0, 0, 1, 1)] * 20,
    }
    
    screenshots = {}
    
    for phase_name, actions in phase_scripts.items():
        obs = env.reset()
        
        for action in actions:
            obs, _, done, _ = env.step(np.array(action, dtype=np.float32))
            if done:
                break
        
        screenshots[phase_name] = obs["rgb"]
    
    env.close()
    
    # Create figure with screenshots
    fig, axes = plt.subplots(1, len(screenshots), figsize=(18, 4))
    
    for ax, (phase_name, img) in zip(axes, screenshots.items()):
        ax.imshow(img)
        ax.set_title(phase_name.upper(), fontsize=12, fontweight='bold')
        ax.axis('off')
    
    fig.suptitle('Environment Visualization at Different Phases', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = out_dir / 'environment_screenshots.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_success_rate_comparison(out_dir):
    """Plot success rate comparison between different methods."""
    print("\n[6/7] Generating success rate comparison chart...")
    
    # Data from your experiments
    methods = ['BC\nBaseline', 'BC + IQL\nCritic', 'Diffusion\n(DDIM)', 'Diffusion\n(Critic-Guided)']
    success_rates = [85.0, 96.67, 35.0, 30.0]  # Approximate values
    colors_map = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(methods, success_rates, color=colors_map, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Add success threshold line
    ax.axhline(y=90, color='green', linestyle='--', linewidth=2, alpha=0.5, label='90% Threshold')
    
    ax.set_ylabel('Success Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('Performance Comparison: Success Rates', fontsize=15, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=11)
    
    # Highlight best method
    bars[1].set_edgecolor('gold')
    bars[1].set_linewidth(4)
    
    plt.tight_layout()
    save_path = out_dir / 'success_rate_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_architecture_diagram(out_dir):
    """Create a simplified architecture diagram."""
    print("\n[7/7] Generating architecture diagram...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define boxes
    boxes = [
        # Input
        {'xy': (0.5, 7), 'width': 1.5, 'height': 1.5, 'color': '#e8f4f8', 'label': 'State\n(9D)', 'fontsize': 11},
        
        # Policy trunk
        {'xy': (3, 6.5), 'width': 2, 'height': 2.5, 'color': '#d4e6f1', 'label': 'Policy Trunk\n(256-dim MLP)', 'fontsize': 10},
        
        # Dual heads
        {'xy': (6, 7.5), 'width': 1.8, 'height': 1.2, 'color': '#aed6f1', 'label': 'Action Head\n(4D)', 'fontsize': 10},
        {'xy': (6, 5.8), 'width': 1.8, 'height': 1.2, 'color': '#aed6f1', 'label': 'Phase Head\n(6-way)', 'fontsize': 10},
        
        # Critic
        {'xy': (3, 3), 'width': 2, 'height': 2, 'color': '#f9e79f', 'label': 'IQL Critic\n(Twin Q-Net)', 'fontsize': 10},
        
        # Action selection
        {'xy': (6.5, 3), 'width': 2.5, 'height': 2, 'color': '#abebc6', 'label': 'Critic-Guided\nAction Selection', 'fontsize': 10},
        
        # Output
        {'xy': (8, 7), 'width': 1.5, 'height': 1.5, 'color': '#c39bd3', 'label': 'Final\nAction', 'fontsize': 11},
    ]
    
    for box in boxes:
        rect = patches.FancyBboxPatch(
            box['xy'], box['width'], box['height'],
            boxstyle="round,pad=0.1", 
            edgecolor='black', facecolor=box['color'],
            linewidth=2
        )
        ax.add_patch(rect)
        
        cx = box['xy'][0] + box['width'] / 2
        cy = box['xy'][1] + box['height'] / 2
        ax.text(cx, cy, box['label'], ha='center', va='center',
                fontsize=box['fontsize'], fontweight='bold')
    
    # Add arrows
    arrows = [
        ((2, 7.75), (3, 7.75)),  # State -> Trunk
        ((5, 8.1), (6, 8.1)),    # Trunk -> Action Head
        ((5, 6.4), (6, 6.4)),    # Trunk -> Phase Head
        ((7.9, 8.1), (8, 8.1)),  # Action Head -> Final
        ((5, 4), (6.5, 4)),      # Critic -> Selection
        ((7.5, 5), (8.5, 7)),    # Selection -> Final
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Add title
    ax.text(5, 9.5, 'Hybrid BC-IQL Architecture', ha='center', va='center',
            fontsize=16, fontweight='bold')
    
    # Add phase info
    phase_text = "Phases: APPROACH → DESCEND → GRASP → LIFT → MOVE → FINE"
    ax.text(5, 0.5, phase_text, ha='center', va='center',
            fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_path = out_dir / 'architecture_diagram.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def main(args):
    """Generate all figures for the report."""
    print("="*60)
    print("GENERATING REPORT FIGURES")
    print("="*60)
    
    out_dir = create_output_dir(args.output_dir)
    
    try:
        # Generate all plots
        plot_phase_distribution(args.data_root, out_dir)
        plot_trajectory_evolution(args.data_root, out_dir, num_episodes=args.num_trajectories)
        plot_action_distributions(args.data_root, out_dir, max_samples=args.max_samples)
        plot_phase_action_heatmap(args.data_root, out_dir, max_samples=args.max_samples)
        capture_environment_screenshots(out_dir)
        plot_success_rate_comparison(out_dir)
        plot_architecture_diagram(out_dir)
        
        print("\n" + "="*60)
        print(f"✓ ALL FIGURES GENERATED SUCCESSFULLY!")
        print(f"✓ Output directory: {out_dir.absolute()}")
        print("="*60)
        
        # List all generated files
        print("\nGenerated files:")
        for file in sorted(out_dir.glob("*.png")):
            print(f"  • {file.name}")
        
    except Exception as e:
        print(f"\n✗ Error generating figures: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate comprehensive figures for experiment report")
    parser.add_argument("--data_root", type=str, default="data/raw/mj_pick_place_v5",
                        help="Path to dataset root directory")
    parser.add_argument("--output_dir", type=str, default="figures",
                        help="Directory to save generated figures")
    parser.add_argument("--num_trajectories", type=int, default=5,
                        help="Number of trajectories to visualize")
    parser.add_argument("--max_samples", type=int, default=10000,
                        help="Maximum samples for action distribution analysis")
    
    args = parser.parse_args()
    main(args)
