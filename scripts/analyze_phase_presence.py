import argparse, numpy as np
from lerobot_dataset.pick_place_mj_builder import MjPickPlaceOfflineDataset
from utils.phase_labeling import label_phase, PHASES

def main(root, horizon):
    ds = MjPickPlaceOfflineDataset(root, use_paraphrase=False)
    counts = {k:0 for k in PHASES.values()}
    seq_counts = {k:0 for k in PHASES.values()}
    for i in range(len(ds)):
        p = label_phase(ds[i]["obs_state"])
        counts[p]+=1
    print("Phase frame counts:", counts)
    print("Fraction:", {k: round(v/len(ds),4) for k,v in counts.items()})
    # sequence-leading phase distribution
    for i in range(0, len(ds)-horizon, horizon):
        p = label_phase(ds[i]["obs_state"])
        seq_counts[p]+=1
    print("Leading sequence phase counts:", seq_counts)

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/raw/mj_pick_place_v5")
    ap.add_argument("--horizon", type=int, default=6)
    a=ap.parse_args()
    main(a.data_root, a.horizon)
