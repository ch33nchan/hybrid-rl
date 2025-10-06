#!/usr/bin/env python3
"""Convenience launcher for diffusion policy training + evaluation on GPU."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from utils.experiment_utils import load_config


def format_float(value: float) -> str:
    if value < 1e-3:
        return f"{value:.0e}"
    return str(value).replace(".", "p").replace("-", "m")


def build_train_command(args):
    cmd = [
        sys.executable,
        "-m",
        "train.train_diffusion_policy_v4",
        "--config",
        args.config,
    ]

    if args.output_dir:
        cmd.extend(["--output_dir", args.output_dir])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.learning_rate is not None:
        cmd.extend(["--learning_rate", str(args.learning_rate)])
    if args.batch_size is not None:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.action_horizon is not None:
        cmd.extend(["--action_horizon", str(args.action_horizon)])
    if args.num_diffusion_steps is not None:
        cmd.extend(["--num_diffusion_steps", str(args.num_diffusion_steps)])
    if args.hidden_dim is not None:
        cmd.extend(["--hidden_dim", str(args.hidden_dim)])
    if args.depth is not None:
        cmd.extend(["--depth", str(args.depth)])
    if args.phase_emb is not None:
        cmd.extend(["--phase_emb", str(args.phase_emb)])
    if args.grad_accum_steps is not None:
        cmd.extend(["--grad_accum_steps", str(args.grad_accum_steps)])
    if args.num_workers is not None:
        cmd.extend(["--num_workers", str(args.num_workers)])
    if args.device:
        cmd.extend(["--device", args.device])
    if args.use_amp:
        cmd.append("--use_amp")
    if args.no_amp:
        cmd.append("--no_amp")

    return cmd


def build_eval_command(args, config, checkpoint_path):
    eval_module = args.eval_script
    cmd = [sys.executable, "-m", eval_module]

    if eval_module == "scripts.evaluate_comprehensive":
        cmd.extend([
            "--config",
            args.config,
            "--method",
            args.method_name,
            "--policy_ckpt",
            checkpoint_path,
        ])
        if args.qnet_ckpt:
            cmd.extend(["--qnet_ckpt", args.qnet_ckpt, "--use_critic"])
    else:
        diffusion_cfg = config.get("diffusion", {})
        ddpm_steps = args.ddim_steps or diffusion_cfg.get("ddim_steps", diffusion_cfg.get("ddim_steps_sweep", [25])[0])
        cmd.extend([
            "--checkpoint",
            checkpoint_path,
            "--episodes",
            str(args.eval_episodes),
            "--seed",
            str(args.eval_seed),
            "--num_steps",
            str(ddpm_steps),
            "--replan_freq",
            str(args.replan_freq or diffusion_cfg.get("replan_freq", 4)),
        ])
        if args.print_every is not None:
            cmd.extend(["--print_every", str(args.print_every)])

    return cmd


def find_checkpoint(config, args):
    diffusion_cfg = config.get("diffusion", {})
    output_dir = Path(args.output_dir or diffusion_cfg.get("output_dir", "models/ckpts_diffusion_cond_v4"))

    if args.checkpoint:
        return Path(args.checkpoint)

    lr = args.learning_rate if args.learning_rate is not None else diffusion_cfg.get("learning_rate", 1e-4)
    batch_size = args.batch_size if args.batch_size is not None else diffusion_cfg.get("batch_size", 64)
    timesteps = args.num_diffusion_steps if args.num_diffusion_steps is not None else diffusion_cfg.get("num_diffusion_steps", diffusion_cfg.get("timesteps", 100))
    horizon = args.action_horizon if args.action_horizon is not None else diffusion_cfg.get("action_horizon", 8)
    experiment_prefix = diffusion_cfg.get("experiment_prefix", "diffusion_policy_v4")

    run_name = f"{experiment_prefix}_lr{format_float(lr)}_bs{batch_size}_T{timesteps}_H{horizon}"
    seed = args.seed if args.seed is not None else diffusion_cfg.get("seed", load_first_seed(config))
    run_dir = output_dir / run_name / f"seed_{seed}"

    candidate = run_dir / "diffusion_policy_v4.pt"
    if candidate.exists():
        return candidate

    # Fallback: pick most recent checkpoint in output_dir
    checkpoints = sorted(output_dir.glob("**/diffusion_policy_v4.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if checkpoints:
        return checkpoints[0]

    raise FileNotFoundError("No diffusion checkpoint found; specify --checkpoint explicitly.")


def load_first_seed(config):
    seeds = config.get("diffusion", {}).get("seed")
    if isinstance(seeds, int):
        return seeds
    global_seeds = config.get("random_seeds", [])
    if global_seeds:
        return global_seeds[0]
    return 0


def main():
    parser = argparse.ArgumentParser(description="GPU orchestration for diffusion policy")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--action_horizon", type=int, default=None)
    parser.add_argument("--num_diffusion_steps", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--train_only", action="store_true", help="Run training only, skip evaluation")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only, skip training")
    parser.add_argument("--checkpoint", type=str, default=None, help="Existing checkpoint to evaluate")
    parser.add_argument("--method_name", type=str, default="diffusion_policy_v4")
    parser.add_argument("--qnet_ckpt", type=str, default=None, help="Critic checkpoint for guided eval")
    parser.add_argument("--eval_script", type=str, default="scripts.eval_diffusion_policy_v4_ddim")
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--eval_seed", type=int, default=2025)
    parser.add_argument("--ddim_steps", type=int, default=None)
    parser.add_argument("--replan_freq", type=int, default=None)
    parser.add_argument("--print_every", type=int, default=10)

    args = parser.parse_args()
    config = load_config(args.config)

    checkpoint_path = None

    if not args.eval_only:
        train_cmd = build_train_command(args)
        print("Launching training:", " ".join(train_cmd))
        subprocess.run(train_cmd, check=True)

    if not args.train_only:
        checkpoint_path = find_checkpoint(config, args)
        eval_cmd = build_eval_command(args, config, str(checkpoint_path))
        print("Launching evaluation:", " ".join(eval_cmd))
        subprocess.run(eval_cmd, check=True)


if __name__ == "__main__":
    main()
