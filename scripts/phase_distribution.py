import argparse, collections
from lerobot_dataset.pick_place_mj_builder import MjPickPlaceOfflineDataset
from utils.phase_labeling import PHASES

def main(root):
    ds = MjPickPlaceOfflineDataset(root, use_paraphrase=False)
    cnt = collections.Counter()
    for i in range(len(ds)):
        cnt[ds[i]["phase_id"]] += 1
    inv = {v:k for k,v in PHASES.items()}
    total = sum(cnt.values())
    print("Total samples:", total)
    for pid in sorted(inv.keys()):
        c = cnt.get(pid,0)
        pct = (c/total*100) if total>0 else 0
        print(f"{pid} ({inv[pid]}): {c} ({pct:.2f}%)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/raw/mj_pick_place_v5")
    args = ap.parse_args()
    main(args.data_root)
