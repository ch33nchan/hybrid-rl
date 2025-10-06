"""
Hyperparameter sweep for BC+IQL (CPU-friendly).
Sweeps critic guidance parameters: num_candidates and noise_scale.
"""

import argparse
import subprocess
import json
from pathlib import Path
from itertools import product
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.experiment_utils import load_config


def run_evaluation(config_path, method_name, policy_ckpt, qnet_ckpt, 
                   num_candidates, noise_scale, results_dir):
    """Run evaluation with specific hyperparameters."""
    
    # Modify config temporarily
    config = load_config(config_path)
    
    # Update candidate generation parameters
    for phase in range(6):
        config['critic_guidance']['num_candidates_per_phase'][phase] = num_candidates
        config['critic_guidance']['noise_scale_per_phase'][phase] = noise_scale
    
    # Save temporary config
    temp_config_path = Path("configs/temp_sweep_config.yaml")
    import yaml
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Run evaluation
    cmd = [
        "python", "-m", "scripts.evaluate_comprehensive",
        "--config", str(temp_config_path),
        "--method", f"{method_name}_K{num_candidates}_noise{noise_scale:.3f}",
        "--policy_ckpt", policy_ckpt,
        "--qnet_ckpt", qnet_ckpt,
        "--use_critic"
    ]
    
    print(f"\nRunning: K={num_candidates}, noise={noise_scale:.3f}")
    result = subprocess.run(cmd, capture_output=False)
    
    # Clean up temp config
    if temp_config_path.exists():
        temp_config_path.unlink()
    
    return result.returncode == 0


def main(args):
    """Run hyperparameter sweep."""
    config = load_config(args.config)
    
    # Get sweep ranges
    num_candidates_sweep = config['critic_guidance']['num_candidates_sweep']
    noise_scale_sweep = config['critic_guidance']['noise_scale_sweep']
    
    print("="*60)
    print("BC+IQL HYPERPARAMETER SWEEP")
    print("="*60)
    print(f"Policy: {args.policy_ckpt}")
    print(f"Critic: {args.qnet_ckpt}")
    print(f"Num candidates sweep: {num_candidates_sweep}")
    print(f"Noise scale sweep: {noise_scale_sweep}")
    print(f"Total combinations: {len(num_candidates_sweep) * len(noise_scale_sweep)}")
    
    results_dir = Path(config['results_dir']) / "sweeps" / "bc_iql"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    sweep_results = []
    
    for num_cand, noise in product(num_candidates_sweep, noise_scale_sweep):
        success = run_evaluation(
            config_path=args.config,
            method_name="bc_iql_sweep",
            policy_ckpt=args.policy_ckpt,
            qnet_ckpt=args.qnet_ckpt,
            num_candidates=num_cand,
            noise_scale=noise,
            results_dir=results_dir
        )
        
        sweep_results.append({
            'num_candidates': num_cand,
            'noise_scale': noise,
            'success': success
        })
    
    # Save sweep summary
    summary_path = results_dir / "sweep_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(sweep_results, f, indent=2)
    
    print(f"\nSweep complete. Summary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml")
    parser.add_argument("--policy_ckpt", type=str, required=True)
    parser.add_argument("--qnet_ckpt", type=str, required=True)
    
    args = parser.parse_args()
    main(args)
