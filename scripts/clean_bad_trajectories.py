import numpy as np, pathlib, os, json
import argparse

def bad(npz):
    try:
        d=np.load(npz)
        for k in ("obs_state","actions"):
            if not np.isfinite(d[k]).all():
                return True
        return False
    except:
        return True

def main(root, delete=False):
    rootp=pathlib.Path(root)
    removed=0
    for ep in rootp.glob("episode_*"):
        npz=ep/"trajectory.npz"
        if not npz.exists(): continue
        if bad(npz):
            print("Corrupt trajectory:", npz)
            if delete:
                for f in ep.glob("*"): 
                    try: f.unlink()
                    except: pass
                try: ep.rmdir()
                except: pass
                removed+=1
    print("Removed" if delete else "Flagged", removed, "episodes")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/raw/mj_pick_place_v5")
    ap.add_argument("--delete", action="store_true")
    a=ap.parse_args()
    main(a.root, delete=a.delete)
