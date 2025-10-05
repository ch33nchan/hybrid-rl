import argparse, torch, numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from lerobot_dataset.pick_place_mj_builder import MjPickPlaceOfflineDataset
from models.multitask_policy import MultiTaskPolicy

def dataset_builder(root):
    base=MjPickPlaceOfflineDataset(root, use_paraphrase=False)
    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(base)
        def __getitem__(self, idx):
            it=base[idx]
            return {
                "state": torch.from_numpy(it["obs_state"].astype(np.float32)),
                "action": torch.from_numpy(it["action"].astype(np.float32)),
                "phase_id": torch.tensor(it["phase_id"], dtype=torch.long)
            }
    return base, DS()

def run_stage(model, loader, device, opt, action_only, phase_coeff, grad_clip):
    model.train()
    losses=[]; bc_ls=[]; ph_ls=[]
    for batch in loader:
        s=batch["state"].to(device)
        a=batch["action"].to(device)
        p=batch["phase_id"].to(device)
        pred_a, logits = model(s)
        l_bc = F.mse_loss(pred_a, a)
        l_ph = F.cross_entropy(logits, p)
        loss = l_bc if action_only else (l_bc + phase_coeff * l_ph)
        opt.zero_grad()
        loss.backward()
        if grad_clip>0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        losses.append(loss.item()); bc_ls.append(l_bc.item()); ph_ls.append(l_ph.item())
    return float(np.mean(losses)), float(np.mean(bc_ls)), float(np.mean(ph_ls))

def main(a):
    base, ds = dataset_builder(a.data_root)
    state_dim= base[0]["obs_state"].shape[0]
    action_dim= base[0]["action"].shape[0]
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True)
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model=MultiTaskPolicy(state_dim, action_dim, num_phases=6, hidden=a.hidden).to(device)
    opt=torch.optim.Adam(model.parameters(), lr=a.lr)
    out=Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for e in range(1,a.stage1_epochs+1):
        tl, bc, ph = run_stage(model, loader, device, opt, action_only=True, phase_coeff=0.0, grad_clip=a.grad_clip)
        print(f"[Stage1 epoch {e}] total={tl:.4f} bc={bc:.4f}")
    torch.save(model.state_dict(), out/"stage1.pt")
    for e in range(1,a.stage2_epochs+1):
        tl, bc, ph = run_stage(model, loader, device, opt, action_only=False, phase_coeff=a.stage2_phase_w, grad_clip=a.grad_clip)
        print(f"[Stage2 epoch {e}] total={tl:.4f} bc={bc:.4f} phase={ph:.4f}")
    torch.save(model.state_dict(), out/"stage2.pt")
    for e in range(1,a.stage3_epochs+1):
        tl, bc, ph = run_stage(model, loader, device, opt, action_only=False, phase_coeff=a.stage3_phase_w, grad_clip=a.grad_clip)
        print(f"[Stage3 epoch {e}] total={tl:.4f} bc={bc:.4f} phase={ph:.4f}")
    torch.save(model.state_dict(), out/"multitask_curriculum_final.pt")
    print("Saved final model:", out/"multitask_curriculum_final.pt")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/raw/mj_pick_place_v5")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--stage1_epochs", type=int, default=2)
    ap.add_argument("--stage2_epochs", type=int, default=2)
    ap.add_argument("--stage3_epochs", type=int, default=4)
    ap.add_argument("--stage2_phase_w", type=float, default=0.1)
    ap.add_argument("--stage3_phase_w", type=float, default=0.25)
    ap.add_argument("--out_dir", type=str, default="models/ckpts_multitask_curriculum")
    a=ap.parse_args()
    main(a)