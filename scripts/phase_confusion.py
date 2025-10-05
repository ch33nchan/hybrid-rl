import argparse, torch, numpy as np
from torch.utils.data import DataLoader
from lerobot_dataset.pick_place_mj_builder import MjPickPlaceOfflineDataset
from models.multitask_policy import MultiTaskPolicy
from utils.phase_labeling import label_phase, PHASES

def main(args):
    ds = MjPickPlaceOfflineDataset(args.data_root, use_paraphrase=False)
    state_dim = ds[0]["obs_state"].shape[0]
    action_dim = ds[0]["action"].shape[0]
    model = MultiTaskPolicy(state_dim, action_dim, num_phases=6)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt["ema"] if isinstance(ckpt, dict) and ckpt.get("ema") else ckpt["model"] if isinstance(ckpt, dict) and ckpt.get("model") else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()

    inv = {v:k for k,v in PHASES.items()}
    conf = np.zeros((6,6), dtype=np.int64)
    sample = min(len(ds), args.max_samples)
    for i in range(sample):
        st = torch.from_numpy(ds[i]["obs_state"]).float().unsqueeze(0)
        with torch.no_grad():
            _, logits = model(st)
        pred = int(torch.argmax(logits, dim=-1).item())
        gt = label_phase(ds[i]["obs_state"])
        conf[gt, pred] += 1

    print("Confusion (rows=GT, cols=Pred):")
    header = "      " + " ".join([f"{inv[c][:4]:>5}" for c in range(6)])
    print(header)
    for r in range(6):
        row = " ".join([f"{conf[r,c]:>5}" for c in range(6)])
        print(f"{inv[r][:6]:>6} {row}")
    print("Per-phase recall:")
    for r in range(6):
        tot = conf[r].sum()
        rec = conf[r,r]/tot if tot>0 else 0
        print(f"{inv[r]}: {rec:.3f} ({tot} samples)")

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/raw/mj_pick_place_v5")
    ap.add_argument("--ckpt", type=str, default="models/ckpts_multitask_balanced_v5/multitask_policy_best.pt")
    ap.add_argument("--max_samples", type=int, default=2000)
    args=ap.parse_args()
    main(args)