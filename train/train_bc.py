import argparse, torch, numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from lerobot_dataset import MjPickPlaceOfflineDataset
from models.bc_policy import SimpleBC

class StateActionDataset(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        it = self.base[idx]
        return {"state": it["obs_state"].astype(np.float32),
                "action": it["action"].astype(np.float32)}

def main(a):
    base = MjPickPlaceOfflineDataset(a.data_root)
    state_dim = base[0]["obs_state"].shape[0]
    action_dim = base[0]["action"].shape[0]
    ds = StateActionDataset(base)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SimpleBC(state_dim, action_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    for ep in range(1, a.epochs+1):
        losses=[]
        for batch in tqdm(dl, desc=f"epoch {ep}"):
            s = batch["state"].to(device)
            act = batch["action"].to(device)
            pred = model(s)
            loss = F.mse_loss(pred, act)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"epoch {ep} loss {sum(losses)/len(losses):.6f}")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out/"bc_policy.pt")
    print("saved", out/"bc_policy.pt")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data/raw/mj_pick_place_v0")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--out_dir", type=str, default="models/ckpts")
    args = p.parse_args()
    main(args)
