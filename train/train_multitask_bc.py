import argparse, torch, numpy as np, math, time, os
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from lerobot_dataset.pick_place_mj_builder import MjPickPlaceOfflineDataset
from models.multitask_policy import MultiTaskPolicy

NUM_PHASES = 6

class MTStateDataset(torch.utils.data.Dataset):
    def __init__(self, base): self.base=base
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        it=self.base[idx]
        return {
            "state": np.asarray(it["obs_state"], dtype=np.float32),
            "action": np.asarray(it["action"], dtype=np.float32),
            "phase_id": int(it["phase_id"])
        }

def compute_phase_counts(ds):
    c = np.zeros(NUM_PHASES, dtype=np.int64)
    for i in range(len(ds)): c[ds[i]["phase_id"]] += 1
    return c

def build_sampler(ds, counts, rare_floor=0.01):
    # Weighted sampling with floor to avoid astronomical scaling
    total = counts.sum()
    freqs = counts / total
    freqs = np.clip(freqs, rare_floor, None)
    inv = 1.0 / freqs
    inv = inv / inv.sum()
    w = np.array([inv[ds[i]["phase_id"]] for i in range(len(ds))])
    w /= w.sum()
    return WeightedRandomSampler(w, num_samples=len(w), replacement=True)

def phase_loss_weights(counts, max_ratio, log_scale=True):
    counts = np.maximum(counts, 1)
    inv = 1.0 / counts
    inv = inv / inv.sum() * NUM_PHASES
    if log_scale:
        inv = np.log(inv + 1.0)
    inv = inv / inv.min()
    inv = np.clip(inv, 1.0, max_ratio)
    inv = inv / inv.mean()
    return inv

def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, device=device)

def moving_average(prev, new, alpha=0.1):
    if prev is None: return new
    return (1-alpha)*prev + alpha*new

def save_checkpoint(model, ema_model, path):
    payload = {
        "model": model.state_dict(),
        "ema": ema_model.state_dict() if ema_model is not None else None
    }
    torch.save(payload, path)

def main(args):
    base = MjPickPlaceOfflineDataset(args.data_root, use_paraphrase=False)
    if len(base)==0:
        raise RuntimeError("Empty dataset")

    state_dim = base[0]["obs_state"].shape[0]
    action_dim= base[0]["action"].shape[0]
    ds = MTStateDataset(base)
    counts = compute_phase_counts(ds)
    print("Phase counts:", counts.tolist())

    if args.sample_balance:
        sampler = build_sampler(ds, counts, rare_floor=0.002)
        loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler)
    else:
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    raw_phase_w = phase_loss_weights(counts, max_ratio=args.weight_cap_ratio, log_scale=True) if args.loss_balance else None

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MultiTaskPolicy(state_dim, action_dim, num_phases=NUM_PHASES, hidden=args.hidden).to(device)
    ema_model = MultiTaskPolicy(state_dim, action_dim, num_phases=NUM_PHASES, hidden=args.hidden).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_decay = args.ema_decay

    # Parameter groups
    phase_params = list(model.phase_head.parameters())
    trunk_params = [p for n,p in model.named_parameters() if not n.startswith("phase_head.")]
    opt_trunk = torch.optim.Adam(trunk_params, lr=args.lr)
    opt_phase = torch.optim.Adam(phase_params, lr=args.lr_phase)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    ema_best_path = out_dir / "multitask_policy_best.pt"
    last_path     = out_dir / "multitask_policy_last.pt"

    phase_w_tensor = torch.tensor(raw_phase_w, dtype=torch.float32, device=device) if raw_phase_w is not None else None
    print("Phase loss weights (effective):", phase_w_tensor.cpu().numpy() if phase_w_tensor is not None else "uniform")

    running_total = None
    grad_explosions = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        t_start = time.time()
        total_losses=[]; bc_losses=[]; ph_losses=[]
        # Dynamic anneal of phase loss weight (small early)
        phase_coeff = args.phase_loss_weight * min(1.0, epoch / args.phase_warmup_epochs)

        for batch in tqdm(loader, desc=f"epoch {epoch}"):
            s = to_device(batch["state"], device)
            a = to_device(batch["action"], device)
            p = torch.as_tensor(batch["phase_id"], device=device, dtype=torch.long)

            pred_a, phase_logits = model(s)
            l_bc = F.mse_loss(pred_a, a)

            if phase_w_tensor is not None:
                ce = F.cross_entropy(phase_logits, p, reduction='none')
                l_ph = (ce * phase_w_tensor[p]).mean()
            else:
                l_ph = F.cross_entropy(phase_logits, p)

            loss = l_bc + phase_coeff * l_ph

            opt_trunk.zero_grad()
            opt_phase.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(trunk_params, args.grad_clip)
            torch.nn.utils.clip_grad_norm_(phase_params, args.grad_clip)

            # Detect exploding gradients (use norm of first trunk parameter)
            with torch.no_grad():
                ref = trunk_params[0].grad
                if ref is not None and torch.isnan(ref).any():
                    grad_explosions += 1
                    continue

            opt_trunk.step()
            opt_phase.step()

            with torch.no_grad():
                for e_p, p_p in zip(ema_model.parameters(), model.parameters()):
                    e_p.data.mul_(1-ema_decay).add_(p_p.data * ema_decay)

            total_losses.append(loss.item()); bc_losses.append(l_bc.item()); ph_losses.append(l_ph.item())
            running_total = moving_average(running_total, loss.item(), alpha=0.05)

        tl = float(np.mean(total_losses)) if total_losses else float('inf')
        msg = f"epoch {epoch} total={tl:.5f} bc={np.mean(bc_losses):.5f} phase={np.mean(ph_losses):.5f} phase_coeff={phase_coeff:.3f} running_avg={running_total:.5f} grad_explosions={grad_explosions}"
        print(msg)

        save_checkpoint(model, ema_model, out_dir / f"multitask_policy_epoch{epoch}.pt")

        if tl < best_loss:
            best_loss = tl
            save_checkpoint(model, ema_model, ema_best_path)
            print(f"Updated best (loss={best_loss:.5f})")

    save_checkpoint(model, ema_model, last_path)
    print("Training complete. Best:", ema_best_path, " Last:", last_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/raw/mj_pick_place_v5")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_phase", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--phase_loss_weight", type=float, default=0.3)
    ap.add_argument("--phase_warmup_epochs", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--sample_balance", action="store_true")
    ap.add_argument("--loss_balance", action="store_true")
    ap.add_argument("--weight_cap_ratio", type=float, default=15.0)
    ap.add_argument("--ema_decay", type=float, default=0.02)
    ap.add_argument("--out_dir", type=str, default="models/ckpts_multitask_balanced_v5")
    args = ap.parse_args()
    main(args)