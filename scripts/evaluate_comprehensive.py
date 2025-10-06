"""
Comprehensive evaluation script with full metrics tracking.
Runs 100 episodes across multiple seeds and test splits.
"""

import argparse
import torch
import numpy as np
import time
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.mj_pick_place_env import MjPickPlaceEnv, MjPickPlaceConfig
from models.multitask_policy import MultiTaskPolicy
from models.critic import QNet
from utils.phase_labeling import label_phase, PHASES
from utils.experiment_utils import (
    set_seed, load_config, get_device, ExperimentLogger,
    EvaluationMetrics, create_test_splits
)

PHASE_NAMES = list(PHASES.keys())


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


def choose_action_with_critic(policy, qnet, state, phase, device, config):
    """Critic-guided action selection with phase-adaptive candidates."""
    with torch.no_grad():
        s_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        base_a, phase_logits = policy(s_t)
        base_a = torch.tanh(base_a).squeeze(0).cpu().numpy()
    
    # Phase-adaptive candidate generation
    num_candidates = config['critic_guidance']['num_candidates_per_phase'].get(phase, 6)
    noise_scale = config['critic_guidance']['noise_scale_per_phase'].get(phase, 0.1)
    
    if num_candidates == 1 or noise_scale == 0.0:
        return base_a, phase_logits.squeeze(0).cpu().numpy()
    
    # Generate candidates
    candidates = [base_a]
    for _ in range(num_candidates - 1):
        noisy = np.clip(base_a + np.random.randn(*base_a.shape) * noise_scale, -1, 1)
        candidates.append(noisy)
    
    # Critic evaluation
    s_b = torch.from_numpy(np.repeat(state[None, :], len(candidates), axis=0)).float().to(device)
    a_b = torch.from_numpy(np.stack(candidates)).float().to(device)
    p_b = torch.full((len(candidates),), phase, dtype=torch.long, device=device)
    
    with torch.no_grad():
        q1, q2 = qnet(s_b, a_b, p_b)
        q_vals = torch.min(q1, q2)
    
    best_idx = int(torch.argmax(q_vals).cpu())
    return candidates[best_idx], phase_logits.squeeze(0).cpu().numpy()


def evaluate_episode(env, policy, qnet, device, config, use_critic=True, 
                     track_phases=False):
    """Evaluate single episode with comprehensive metrics."""
    obs = env.reset()
    
    steps = 0
    done = False
    start_time = time.time()
    
    # Tracking
    collisions = 0
    grasp_attempted = False
    grasp_success = False
    phases_encountered = []
    phase_predictions = []
    phase_targets = []
    
    while not done and steps < env.cfg.max_steps:
        state = obs["state"]
        true_phase = label_phase(state)
        phases_encountered.append(true_phase)
        
        # Get action
        if use_critic and qnet is not None:
            action, phase_logits = choose_action_with_critic(
                policy, qnet, state, true_phase, device, config
            )
            pred_phase = int(np.argmax(phase_logits))
        else:
            with torch.no_grad():
                s_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
                a_t, phase_logits = policy(s_t)
                action = torch.tanh(a_t).squeeze(0).cpu().numpy()
                pred_phase = int(torch.argmax(phase_logits).cpu())
        
        if track_phases:
            phase_predictions.append(pred_phase)
            phase_targets.append(true_phase)
        
        # Track grasp attempts
        if true_phase == PHASES["GRASP_SETTLE"] and not grasp_attempted:
            grasp_attempted = True
            if action[3] > 0.5:  # Gripper closing
                grasp_success = True
        
        # Step environment
        obs, reward, done, info = env.step(action)
        steps += 1
    
    episode_time = time.time() - start_time
    success = info.get("success", False)
    
    # Determine failure phase if failed
    failure_phase = None
    if not success and phases_encountered:
        failure_phase = phases_encountered[-1]
    
    results = {
        'success': success,
        'steps': steps,
        'time': episode_time,
        'collisions': collisions,
        'grasp_success': grasp_success,
        'failure_phase': failure_phase,
    }
    
    if track_phases:
        results['phase_predictions'] = phase_predictions
        results['phase_targets'] = phase_targets
    
    return results


def evaluate_split(env_config, policy, qnet, device, config, split_config, 
                   num_episodes, use_critic=True, seed=42):
    """Evaluate on a specific test split."""
    set_seed(seed)
    
    # Modify environment config for this split
    env_cfg = MjPickPlaceConfig(
        seed=seed,
        xml_path=env_config['xml_path'],
        max_steps=env_config['max_steps']
    )
    
    env = MjPickPlaceEnv(env_cfg)
    metrics = EvaluationMetrics()
    
    all_phase_preds = []
    all_phase_targets = []
    
    for ep in tqdm(range(num_episodes), desc=f"Eval {split_config['name']}"):
        # Set episode seed for reproducibility
        env.cfg.seed = seed + ep
        
        results = evaluate_episode(
            env, policy, qnet, device, config, use_critic=use_critic,
            track_phases=(ep < 10)  # Track phases for first 10 episodes
        )
        
        metrics.add_episode(
            success=results['success'],
            steps=results['steps'],
            time=results['time'],
            collisions=results['collisions'],
            grasp_success=results['grasp_success'],
            failure_phase=results['failure_phase']
        )
        
        if 'phase_predictions' in results:
            all_phase_preds.extend(results['phase_predictions'])
            all_phase_targets.extend(results['phase_targets'])
    
    env.close()
    
    stats = metrics.compute_statistics()
    stats['split_name'] = split_config['name']
    stats['split_description'] = split_config['description']
    
    # Phase prediction accuracy
    if all_phase_preds:
        stats['phase_prediction_accuracy'] = float(
            (np.array(all_phase_preds) == np.array(all_phase_targets)).mean()
        )
    
    return stats, all_phase_preds, all_phase_targets


def main(args):
    """Main evaluation loop."""
    # Load configuration
    config = load_config(args.config)
    device = get_device(config['device'])
    
    print(f"Device: {device}")
    print(f"Evaluating: {args.method}")
    print(f"Seeds: {config['random_seeds']}")
    print(f"Episodes per seed: {config['evaluation']['num_episodes']}")
    
    # Create test splits
    test_splits = create_test_splits(
        num_splits=config['evaluation']['num_test_splits'],
        seed=42
    )
    
    # Initialize environment to get dimensions
    env_tmp = MjPickPlaceEnv(MjPickPlaceConfig(seed=42))
    probe = env_tmp.reset()
    state_dim = probe["state"].shape[0]
    action_dim = 4
    env_tmp.close()
    
    # Load models
    print("\nLoading models...")
    policy = load_policy(args.policy_ckpt, state_dim, action_dim, device)
    qnet = None
    if args.qnet_ckpt and args.use_critic:
        qnet = load_critic(args.qnet_ckpt, state_dim, action_dim, device)
    print("Models loaded")
    
    # Results storage
    all_seed_results = []
    
    # Evaluate across seeds
    for seed_idx, seed in enumerate(config['random_seeds']):
        print(f"\n{'='*60}")
        print(f"Seed {seed_idx + 1}/{len(config['random_seeds'])}: {seed}")
        print(f"{'='*60}")
        
        set_seed(seed, deterministic=config['reproducibility']['deterministic'])
        
        # Create logger
        logger = ExperimentLogger(
            config['log_dir'],
            f"{args.method}_eval",
            seed
        )
        logger.log_config({
            'method': args.method,
            'policy_ckpt': args.policy_ckpt,
            'qnet_ckpt': args.qnet_ckpt,
            'use_critic': args.use_critic,
            'seed': seed,
            'config': config
        })
        
        seed_results = {'seed': seed, 'splits': []}
        
        # Evaluate on each test split
        for split in test_splits:
            print(f"\nEvaluating on split: {split['name']}")
            
            stats, phase_preds, phase_targets = evaluate_split(
                env_config=config['environment'],
                policy=policy,
                qnet=qnet,
                device=device,
                config=config,
                split_config=split,
                num_episodes=config['evaluation']['num_episodes'],
                use_critic=args.use_critic,
                seed=seed
            )
            
            seed_results['splits'].append(stats)
            
            # Print results
            print(f"\nResults for {split['name']}:")
            print(f"  Success Rate: {stats['success_rate_mean']:.4f} +/- {stats['success_rate_ci_95_upper'] - stats['success_rate_mean']:.4f}")
            print(f"  Avg Steps: {stats['avg_steps_mean']:.2f} +/- {stats['avg_steps_std']:.2f}")
            print(f"  Avg Time: {stats['avg_time_mean']:.3f}s +/- {stats['avg_time_std']:.3f}s")
            print(f"  Grasp Success: {stats['grasp_success_rate']:.4f}")
            
            if 'failures_by_phase' in stats:
                print(f"  Failures by phase: {stats['failures_by_phase']}")
            
            # Save confusion matrix for phase predictions
            if phase_preds and phase_targets:
                from utils.experiment_utils import save_confusion_matrix
                cm_path = logger.log_dir / f"phase_confusion_{split['name']}.json"
                save_confusion_matrix(
                    np.array(phase_preds),
                    np.array(phase_targets),
                    str(cm_path),
                    PHASE_NAMES
                )
        
        # Aggregate across splits for this seed
        split_success_rates = [s['success_rate_mean'] for s in seed_results['splits']]
        seed_results['mean_success_across_splits'] = float(np.mean(split_success_rates))
        seed_results['std_success_across_splits'] = float(np.std(split_success_rates))
        
        logger.log_results(seed_results)
        all_seed_results.append(seed_results)
    
    # Aggregate across all seeds
    print(f"\n{'='*60}")
    print("FINAL RESULTS ACROSS ALL SEEDS")
    print(f"{'='*60}")
    
    # For each split, aggregate across seeds
    for split_idx, split in enumerate(test_splits):
        split_name = split['name']
        split_results = [sr['splits'][split_idx] for sr in all_seed_results]
        
        success_rates = [r['success_rate_mean'] for r in split_results]
        mean_sr = np.mean(success_rates)
        std_sr = np.std(success_rates)
        se_sr = std_sr / np.sqrt(len(success_rates))
        ci_95 = 1.96 * se_sr
        
        print(f"\nSplit: {split_name}")
        print(f"  Success Rate: {mean_sr:.4f} +/- {ci_95:.4f} (95% CI)")
        print(f"  Std Dev: {std_sr:.4f}")
        print(f"  All seeds: {[f'{sr:.4f}' for sr in success_rates]}")
    
    # Save aggregated results
    final_results = {
        'method': args.method,
        'num_seeds': len(config['random_seeds']),
        'episodes_per_seed': config['evaluation']['num_episodes'],
        'total_episodes': len(config['random_seeds']) * config['evaluation']['num_episodes'],
        'all_seed_results': all_seed_results,
    }
    
    results_path = Path(config['results_dir']) / f"{args.method}_comprehensive_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive evaluation with statistical analysis")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml")
    parser.add_argument("--method", type=str, required=True, help="Method name (e.g., bc_critic)")
    parser.add_argument("--policy_ckpt", type=str, required=True)
    parser.add_argument("--qnet_ckpt", type=str, default=None)
    parser.add_argument("--use_critic", action="store_true", default=False)
    
    args = parser.parse_args()
    main(args)
