"""GPU-ready training entrypoint for the phase-conditioned diffusion policy."""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from lerobot_dataset.pick_place_mj_builder import MjPickPlaceOfflineDataset
from utils.phase_labeling import label_phase
from utils.experiment_utils import (
    ExperimentLogger,
    get_device,
    load_config,
    set_seed,
)


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-5, 0.02)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        if half > 0:
            freqs = torch.exp(
                -torch.arange(half, device=device)
                * (torch.log(torch.tensor(10000.0, device=device)) / (half - 1 + 1e-8))
            )
            args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
            emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        else:
            emb = torch.zeros((t.shape[0], 0), device=device)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=device)], dim=-1)
        return emb


class ResBlock(nn.Module):
    def __init__(self, dim: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_block = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.LayerNorm(dim))
        self.time_proj = nn.Linear(time_dim, dim)
        self.out_block = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.in_block(x)
        h = h + self.time_proj(t_emb)
        return self.out_block(h) + x


class CondDiffusionNetV4(nn.Module):
    def __init__(
        self,
        seq_dim: int,
        state_dim: int,
        *,
        phase_emb: int = 16,
        model_dim: int = 512,
        time_dim: int = 128,
        depth: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, model_dim),
            nn.SiLU(),
        )
        self.phase_emb = nn.Embedding(6, phase_emb)
        self.in_proj = nn.Linear(seq_dim + state_dim + phase_emb, model_dim)
        self.blocks = nn.ModuleList(
            [ResBlock(model_dim, model_dim, dropout=dropout) for _ in range(depth)]
        )
        self.out = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2), nn.SiLU(), nn.Linear(model_dim // 2, seq_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        phase: torch.Tensor,
        t: torch.Tensor,
        drop_prob: float = 0.1,
    ) -> torch.Tensor:
        if self.training and drop_prob > 0:
            mask = (torch.rand_like(state[:, 0]) < drop_prob).float().unsqueeze(1)
            state = state * (1 - mask)
        temb = self.time_mlp(t)
        pemb = self.phase_emb(phase)
        h = torch.cat([x, state, pemb], dim=-1)
        h = self.in_proj(h)
        for block in self.blocks:
            h = block(h, temb)
        return self.out(h)


def build_sequences(base_dataset, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    seqs, conds, phases = [], [], []
    for i in range(len(base_dataset) - horizon):
        ep_i, _ = base_dataset.index[i]
        ep_j, _ = base_dataset.index[i + horizon - 1]
        if ep_i != ep_j:
            continue
        s0 = base_dataset[i]["obs_state"]
        acts = [base_dataset[i + k]["action"] for k in range(horizon)]
        seqs.append(np.concatenate(acts))
        conds.append(s0)
        phases.append(label_phase(s0))
    return (
        np.asarray(seqs, dtype=np.float32),
        np.asarray(conds, dtype=np.float32),
        np.asarray(phases, dtype=np.int64),
    )


def normalize(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = data.mean(0, keepdims=True)
    std = data.std(0, keepdims=True) + 1e-6
    return (data - mean) / std, mean.squeeze(0), std.squeeze(0)


class DiffusionSequenceDataset(Dataset):
    def __init__(self, sequences, states, phases):
        self.sequences = torch.from_numpy(sequences)
        self.states = torch.from_numpy(states)
        self.phases = torch.from_numpy(phases)

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "sequence": self.sequences[idx],
            "state": self.states[idx],
            "phase": self.phases[idx],
        }


def create_dataloader(dataset: DiffusionSequenceDataset, batch_size: int, num_workers: int, device: torch.device) -> DataLoader:
    pin_memory = device.type == "cuda"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def format_float(value: float) -> str:
    return f"{value:.0e}" if value < 1e-3 else str(value).replace(".", "p").replace("-", "m")


def main(args: argparse.Namespace) -> None:
    config = load_config(args.config) if args.config else {}
    diffusion_cfg = config.get("diffusion", {})
    dataset_cfg = config.get("dataset", {})
    reproducibility_cfg = config.get("reproducibility", {})

    seed = args.seed if args.seed is not None else diffusion_cfg.get("seed", config.get("random_seeds", [0])[0])
    set_seed(seed, deterministic=reproducibility_cfg.get("deterministic", True))

    if args.device:
        device = get_device(args.device)
    else:
        device = get_device("cuda" if torch.cuda.is_available() else diffusion_cfg.get("device", "auto"))

    data_root = dataset_cfg.get("root", "data/raw/mj_pick_place_v5")
    horizon = args.action_horizon or diffusion_cfg.get("action_horizon", 8)
    timesteps = args.num_diffusion_steps or diffusion_cfg.get("num_diffusion_steps", diffusion_cfg.get("timesteps", 100))
    hidden_dim = args.hidden_dim or diffusion_cfg.get("model_dim", 512)
    depth = args.depth or diffusion_cfg.get("depth", 4)
    phase_emb = args.phase_emb or diffusion_cfg.get("phase_emb", 16)
    dropout = diffusion_cfg.get("dropout", 0.1)

    batch_size = args.batch_size or diffusion_cfg.get("batch_size", 128)
    learning_rate = args.learning_rate or diffusion_cfg.get("learning_rate", 3e-4)
    weight_decay = diffusion_cfg.get("weight_decay", 1e-4)
    epochs = args.epochs or diffusion_cfg.get("epochs", 50)
    cond_dropout = diffusion_cfg.get("cond_dropout", 0.1)
    gradient_accumulation_steps = args.grad_accum_steps or diffusion_cfg.get("gradient_accumulation_steps", 1)
    num_workers = args.num_workers if args.num_workers is not None else diffusion_cfg.get("num_workers", 4)
    use_amp = diffusion_cfg.get("use_amp", False)
    if args.use_amp:
        use_amp = True
    if args.no_amp:
        use_amp = False

    experiment_prefix = diffusion_cfg.get("experiment_prefix", "diffusion_policy_v4")
    run_name = (
        f"{experiment_prefix}_lr{format_float(learning_rate)}_bs{batch_size}_T{timesteps}_H{horizon}"
    )
    output_dir = Path(args.output_dir or diffusion_cfg.get("output_dir", "models/ckpts_diffusion_cond_v4"))
    run_dir = output_dir / run_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = ExperimentLogger(
        log_dir=config.get("log_dir", "logs"),
        experiment_name=run_name,
        seed=seed,
    )
    logger.log_config(
        {
            "seed": seed,
            "device": str(device),
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "timesteps": timesteps,
            "horizon": horizon,
            "hidden_dim": hidden_dim,
            "depth": depth,
            "phase_emb": phase_emb,
            "epochs": epochs,
            "cond_dropout": cond_dropout,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "use_amp": use_amp,
            "num_workers": num_workers,
            "dataset_root": data_root,
        }
    )

    dataset = MjPickPlaceOfflineDataset(
        data_root,
        use_paraphrase=dataset_cfg.get("use_paraphrase", False),
        max_samples=dataset_cfg.get("max_samples"),
    )
    sequences, conds, phases = build_sequences(dataset, horizon)
    if len(sequences) == 0:
        raise RuntimeError("No training sequences were constructed. Check the dataset and horizon.")

    seqs_norm, act_mean, act_std = normalize(sequences)
    states_norm, state_mean, state_std = normalize(conds)
    train_dataset = DiffusionSequenceDataset(seqs_norm, states_norm, phases)
    dataloader = create_dataloader(train_dataset, batch_size, num_workers, device)

    seq_dim = seqs_norm.shape[1]
    state_dim = states_norm.shape[1]
    model = CondDiffusionNetV4(
        seq_dim,
        state_dim,
        phase_emb=phase_emb,
        model_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
    ).to(device)

    betas = cosine_beta_schedule(timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cum = torch.cumprod(alphas, dim=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    global_step = 0
    steps_per_epoch = len(dataloader)
    total_batches = epochs * steps_per_epoch
    overall_progress = tqdm(total=total_batches, desc="training", unit="batch")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        optimizer.zero_grad(set_to_none=True)

        with tqdm(dataloader, desc=f"epoch {epoch}/{epochs}", leave=False, unit="batch") as epoch_bar:
            for step, batch in enumerate(epoch_bar):
                clean = batch["sequence"].to(device)
                state_b = batch["state"].to(device)
                phase_b = batch["phase"].to(device)

                t = torch.randint(0, timesteps, (clean.size(0),), device=device)
                alpha_bar = alphas_cum[t].unsqueeze(1)
                noise = torch.randn_like(clean)
                noisy = torch.sqrt(alpha_bar) * clean + torch.sqrt(1 - alpha_bar) * noise

                with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                    pred_noise = model(noisy, state_b, phase_b, t, drop_prob=cond_dropout)
                    loss = F.mse_loss(pred_noise, noise)

                loss_to_backward = loss / gradient_accumulation_steps
                scaler.scale(loss_to_backward).backward()

                should_step = (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(dataloader)
                if should_step:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                epoch_losses.append(loss.item())
                global_step += 1
                epoch_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                overall_progress.update(1)
                overall_progress.set_postfix(
                    {"epoch": f"{epoch}/{epochs}", "loss": f"{loss.item():.4f}"}
                )

        scheduler.step()
        mean_loss = float(np.mean(epoch_losses))
        logger.log_metric(
            epoch,
            {
                "epoch": epoch,
                "loss": mean_loss,
                "lr": scheduler.get_last_lr()[0],
                "steps": global_step,
            },
        )

    overall_progress.close()

    checkpoint = {
        "model": model.state_dict(),
        "seq_dim": seq_dim,
        "state_dim": state_dim,
        "horizon": horizon,
        "timesteps": timesteps,
        "betas": betas.detach().cpu(),
        "act_mean": act_mean,
        "act_std": act_std,
        "state_mean": state_mean,
        "state_std": state_std,
        "hidden_dim": hidden_dim,
        "depth": depth,
        "phase_emb": phase_emb,
        "cond_dropout": cond_dropout,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "use_amp": use_amp,
        "device": str(device),
        "run_name": run_name,
        "seed": seed,
    }

    ckpt_path = run_dir / "diffusion_policy_v4.pt"
    torch.save(checkpoint, ckpt_path)

    stats_path = run_dir / "training_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "loss": mean_loss,
                "epochs": epochs,
                "num_samples": len(train_dataset),
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "gradient_accumulation_steps": gradient_accumulation_steps,
            },
            indent=2,
        )
    )

    logger.log_results({"final_loss": mean_loss, "checkpoint_path": str(ckpt_path)})
    print(f"Saved checkpoint to {ckpt_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the phase-conditioned diffusion policy (GPU-ready).")
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
    parser.add_argument("--phase_emb", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--use_amp", action="store_true", help="Force-enable AMP even if config disables it.")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP regardless of config.")
    parser.add_argument("--device", type=str, default=None, help="Override device selection (e.g. 'cuda').")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
