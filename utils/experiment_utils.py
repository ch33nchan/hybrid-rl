"""
Experiment utilities for reproducibility, logging, and statistical analysis.
"""

import numpy as np
import torch
import random
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import yaml


def set_seed(seed: int, deterministic: bool = True):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict:
    """Load experiment configuration from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_device(device_str: str = "auto") -> torch.device:
    """Get compute device with auto-detection."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


class ExperimentLogger:
    """Structured logging for experiments."""
    
    def __init__(self, log_dir: str, experiment_name: str, seed: int):
        self.log_dir = Path(log_dir) / experiment_name / f"seed_{seed}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.log_dir / "metrics.csv"
        self.config_file = self.log_dir / "config.json"
        self.results_file = self.log_dir / "results.json"
        
        self.metrics = []
        
    def log_config(self, config: Dict):
        """Save experiment configuration."""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def log_metric(self, step: int, metrics: Dict[str, float]):
        """Log metrics at a given step."""
        metrics['step'] = step
        metrics['timestamp'] = datetime.now().isoformat()
        self.metrics.append(metrics)
        
        # Append to CSV
        if not self.metrics_file.exists():
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=metrics.keys())
                writer.writeheader()
                writer.writerow(metrics)
        else:
            with open(self.metrics_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=metrics.keys())
                writer.writerow(metrics)
    
    def log_results(self, results: Dict):
        """Save final results."""
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def get_metrics(self) -> List[Dict]:
        """Get all logged metrics."""
        return self.metrics


class EvaluationMetrics:
    """Track and compute evaluation metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.episodes = []
        self.successes = []
        self.steps = []
        self.times = []
        self.collisions = []
        self.grasp_successes = []
        self.failure_phases = []
    
    def add_episode(self, success: bool, steps: int, time: float,
                   collisions: int = 0, grasp_success: bool = False,
                   failure_phase: int = None):
        """Add episode results."""
        self.episodes.append(len(self.episodes))
        self.successes.append(success)
        self.steps.append(steps)
        self.times.append(time)
        self.collisions.append(collisions)
        self.grasp_successes.append(grasp_success)
        if not success and failure_phase is not None:
            self.failure_phases.append(failure_phase)
    
    def compute_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        successes = np.array(self.successes, dtype=float)
        steps = np.array(self.steps, dtype=float)
        times = np.array(self.times, dtype=float)
        
        n = len(successes)
        success_rate = successes.mean()
        success_std = successes.std()
        success_se = success_std / np.sqrt(n)
        
        # 95% confidence interval
        success_ci_95 = 1.96 * success_se
        
        stats = {
            'num_episodes': n,
            'success_rate_mean': float(success_rate),
            'success_rate_std': float(success_std),
            'success_rate_se': float(success_se),
            'success_rate_ci_95_lower': float(max(0, success_rate - success_ci_95)),
            'success_rate_ci_95_upper': float(min(1, success_rate + success_ci_95)),
            'avg_steps_mean': float(steps.mean()),
            'avg_steps_std': float(steps.std()),
            'avg_time_mean': float(times.mean()),
            'avg_time_std': float(times.std()),
            'avg_collisions': float(np.mean(self.collisions)),
            'grasp_success_rate': float(np.mean(self.grasp_successes)),
        }
        
        # Failure analysis by phase
        if self.failure_phases:
            phase_counts = np.bincount(self.failure_phases, minlength=6)
            stats['failures_by_phase'] = phase_counts.tolist()
        
        return stats


def aggregate_seed_results(results_list: List[Dict]) -> Dict[str, Any]:
    """Aggregate results across multiple seeds."""
    
    # Extract success rates
    success_rates = [r['success_rate_mean'] for r in results_list]
    
    aggregated = {
        'num_seeds': len(results_list),
        'success_rate_mean': float(np.mean(success_rates)),
        'success_rate_std': float(np.std(success_rates)),
        'success_rate_se': float(np.std(success_rates) / np.sqrt(len(success_rates))),
        'success_rate_ci_95_lower': float(np.mean(success_rates) - 1.96 * np.std(success_rates) / np.sqrt(len(success_rates))),
        'success_rate_ci_95_upper': float(np.mean(success_rates) + 1.96 * np.std(success_rates) / np.sqrt(len(success_rates))),
        'success_rates_all_seeds': success_rates,
    }
    
    # Aggregate other metrics
    for key in ['avg_steps_mean', 'avg_time_mean', 'avg_collisions', 'grasp_success_rate']:
        if key in results_list[0]:
            values = [r[key] for r in results_list]
            aggregated[f'{key}_across_seeds'] = float(np.mean(values))
            aggregated[f'{key}_std_across_seeds'] = float(np.std(values))
    
    return aggregated


def create_test_splits(num_splits: int = 3, seed: int = 42) -> List[Dict]:
    """Create different test environment configurations for OOD evaluation."""
    rng = np.random.RandomState(seed)
    
    splits = []
    
    # Split 1: Standard (in-distribution)
    splits.append({
        'name': 'standard',
        'cube_range': [-0.08, 0.08],
        'target_range': [-0.08, 0.08],
        'description': 'In-distribution positions'
    })
    
    # Split 2: Extended range (mild OOD)
    splits.append({
        'name': 'extended',
        'cube_range': [-0.10, 0.10],
        'target_range': [-0.10, 0.10],
        'description': 'Extended position range'
    })
    
    # Split 3: Corner cases (hard OOD)
    splits.append({
        'name': 'corners',
        'cube_range': [-0.09, 0.09],
        'target_range': [-0.09, 0.09],
        'description': 'Corner positions (hard cases)',
        'force_corners': True
    })
    
    return splits[:num_splits]


def save_confusion_matrix(predictions: np.ndarray, targets: np.ndarray, 
                         save_path: str, phase_names: List[str]):
    """Save phase prediction confusion matrix."""
    from sklearn.metrics import confusion_matrix
    
    num_phases = len(phase_names)

    if len(targets) == 0:
        cm = np.zeros((num_phases, num_phases), dtype=int)
    else:
        cm = confusion_matrix(targets, predictions, labels=list(range(num_phases)))
    
    # Save as JSON
    cm_dict = {
        'confusion_matrix': cm.tolist(),
        'phase_names': phase_names,
        'accuracy': float((predictions == targets).mean()),
        'per_phase_accuracy': []
    }
    
    for i in range(len(phase_names)):
        if cm[i].sum() > 0:
            acc = cm[i, i] / cm[i].sum()
        else:
            acc = 0.0
        cm_dict['per_phase_accuracy'].append({
            'phase': phase_names[i],
            'accuracy': float(acc)
        })
    
    with open(save_path, 'w') as f:
        json.dump(cm_dict, f, indent=2)
    
    return cm_dict
