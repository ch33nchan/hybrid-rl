import argparse, torch, numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from lerobot_dataset.pick_place_mj_builder import MjPickPlaceOfflineDataset
from models.critic import ValueNet, QNet

NUM_PHASES=6

class TransitionDataset(torch.utils.data.Dataset):
    def __init__(self, base, discount=0.99, shaped=False, progress=False):
        self.base = base
        self.discount = discount
        self.shaped = shaped
        self.progress = progress
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        it = self.base[idx]
        s  = it["obs_state"].astype(np.float32)
        ns = it["next_obs_state"].astype(np.float32)
        a  = it["action"].astype(np.float32)
        done  = float(it["done"])
        success = float(it["success"])
        r = success if done and success > 0 else 0.0
        if self.shaped:
            cube = s[4:7]; tgt = s[7:9]
            r += -0.02 * np.linalg.norm(cube[:2]-tgt)
        if self.progress:
            cube = s[4:7]; tgt = s[7:9]; n_cube = ns[4:7]
            r += 0.08 * (np.linalg.norm(cube[:2]-tgt) - np.linalg.norm(n_cube[:2]-tgt))
        return {"s":s,"a":a,"ns":ns,"r":np.float32(r),"d":np.float32(done),"phase":it["phase_id"]}

def expectile_loss(diff, expectile):
    w = torch.where(diff>0, expectile, 1-expectile)
    return (w*diff.pow(2)).mean()

def compute_phase_weights(base):
    c = np.zeros(NUM_PHASES,dtype=np.int64)
    for i in range(len(base)):
        c[base[i]["phase_id"]] += 1
    c=np.maximum(c,1)
    inv=1.0/c
    w=inv/inv.sum()*NUM_PHASES
    return torch.tensor(w,dtype=torch.float32)

def to_t(x,d):
    if isinstance(x,torch.Tensor): return x.to(d)
    return torch.as_tensor(x,device=d)

def main(a):
    base = MjPickPlaceOfflineDataset(a.data_root, use_paraphrase=False)
    ds = TransitionDataset(base, discount=a.gamma, shaped=a.shaped_reward, progress=a.progress_reward)
    loader = DataLoader(ds,batch_size=a.batch_size,shuffle=True,drop_last=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    sdim = base[0]["obs_state"].shape[0]; adim=base[0]["action"].shape[0]

    vnet = ValueNet(sdim).to(device)
    vnet_target = ValueNet(sdim).to(device)
    vnet_target.load_state_dict(vnet.state_dict())

    qnet = QNet(sdim, adim, twin=True).to(device)

    opt_v = torch.optim.Adam(vnet.parameters(), lr=a.lr)
    opt_q = torch.optim.Adam(qnet.parameters(), lr=a.lr)

    phase_w = compute_phase_weights(base).to(device) if a.phase_balance else None
    if phase_w is not None: print("Phase weights:", phase_w.cpu().numpy())

    for epoch in range(1,a.epochs+1):
        ql=[]; vl=[]
        for batch in tqdm(loader, desc=f"epoch {epoch}"):
            s  = to_t(batch["s"],device)
            a_act = to_t(batch["a"],device)
            ns = to_t(batch["ns"],device)
            r  = to_t(batch["r"],device)
            d  = to_t(batch["d"],device)
            p  = torch.as_tensor(batch["phase"],device=device,dtype=torch.long)

          
            noise = torch.randn_like(a_act)*0.05
            a_noisy = torch.clamp(a_act + noise, -1,1)

            with torch.no_grad():
                v_ns = vnet_target(ns,p)
                target_q = r + a.gamma * (1 - d) * v_ns

            q1, q2 = qnet(s,a_noisy,p)
            q_min = torch.min(q1,q2)
            if phase_w is not None:
                qw = phase_w[p]
                loss_q = ((q1-target_q).pow(2)+(q2-target_q).pow(2))*0.5
                loss_q = (loss_q * qw).mean()
            else:
                loss_q = 0.5*((q1-target_q).pow(2)+(q2-target_q).pow(2)).mean()
            opt_q.zero_grad(); loss_q.backward()
            if a.grad_clip>0: torch.nn.utils.clip_grad_norm_(qnet.parameters(), a.grad_clip)
            opt_q.step()

            with torch.no_grad():
                q1_det,q2_det = qnet(s,a_act,p)
                q_min_det = torch.min(q1_det,q2_det)

            diff = q_min_det - vnet(s,p)
            loss_v = expectile_loss(diff, a.expectile)
            opt_v.zero_grad(); loss_v.backward()
            if a.grad_clip>0: torch.nn.utils.clip_grad_norm_(vnet.parameters(), a.grad_clip)
            opt_v.step()

            
            with torch.no_grad():
                for tp, sp in zip(vnet_target.parameters(), vnet.parameters()):
                    tp.data.mul_(1 - a.ema_tau).add_(sp.data * a.ema_tau)

            ql.append(loss_q.item()); vl.append(loss_v.item())

        print(f"epoch {epoch} q_loss={np.mean(ql):.6f} v_loss={np.mean(vl):.6f}")
        out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
        torch.save(qnet.state_dict(), out/"qnet.pt")
        torch.save(vnet.state_dict(), out/"vnet.pt")
        torch.save(vnet_target.state_dict(), out/"vnet_target.pt")
        print("saved critics epoch", epoch)

if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/raw/mj_pick_place_v5")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--expectile", type=float, default=0.7)
    ap.add_argument("--ema_tau", type=float, default=0.02)
    ap.add_argument("--shaped_reward", action="store_true")
    ap.add_argument("--progress_reward", action="store_true")
    ap.add_argument("--phase_balance", action="store_true")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--out_dir", type=str, default="models/ckpts_iql_balanced_v2")
    a=ap.parse_args()
    main(a)
