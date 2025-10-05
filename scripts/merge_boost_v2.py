import argparse, pathlib, shutil

def main(src, dst):
    srcp=pathlib.Path(src); dstp=pathlib.Path(dst)
    existing = sorted([p for p in dstp.glob("episode_*") if p.is_dir()])
    offset = len(existing)
    i=0
    for ep in sorted(srcp.glob("*")):
        if not ep.is_dir(): continue
        new = dstp / f"episode_{offset+i:03d}"
        shutil.copytree(ep, new, dirs_exist_ok=True)
        i+=1
    print("Merged", i, "episodes. Total now:", offset+i)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--boost_src", type=str, default="data/raw/mj_pick_place_boost_v2")
    ap.add_argument("--main_dst", type=str, default="data/raw/mj_pick_place_v5")
    a=ap.parse_args()
    main(a.boost_src, a.main_dst)