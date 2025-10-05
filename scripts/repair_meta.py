import argparse, pathlib, json, os, tempfile, shutil

def is_valid_json(p: pathlib.Path):
    try:
        with open(p,"r") as f:
            json.load(f)
        return True
    except:
        return False

def safe_write(p: pathlib.Path, obj):
    tmp = p.parent / (".repair_"+p.name)
    with open(tmp,"w") as f:
        json.dump(obj,f,indent=2)
    os.replace(tmp,p)

def repair(root, delete=False):
    rootp=pathlib.Path(root)
    bad=[]
    for meta in rootp.rglob("meta.json"):
        if not is_valid_json(meta):
            bad.append(meta)
    for b in bad:
        if delete:
            print("Deleting corrupt meta:", b)
            b.unlink(missing_ok=True)
        else:
            print("Repairing meta:", b)
            safe_write(b, {
                "type":"repaired",
                "success": False,
                "timestamp":"REPAIRED",
                "note":"Corrupt meta replaced"
            })
    print("Total corrupt meta handled:", len(bad))

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/raw/mj_pick_place_v5")
    ap.add_argument("--delete", action="store_true")
    a=ap.parse_args()
    repair(a.root, delete=a.delete)
