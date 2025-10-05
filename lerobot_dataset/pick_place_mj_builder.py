import numpy as np
from pathlib import Path
import json, random, sys
from utils.phase_labeling import label_phase

class MjPickPlaceOfflineDataset:
    def __init__(self, root: str, use_paraphrase: bool = True, seed: int = 0, verbose: bool = True):
        self.root = Path(root)
        self.use_paraphrase = use_paraphrase
        self.rng = random.Random(seed)
        self.episodes = sorted([d for d in self.root.glob("episode_*") if d.is_dir()])
        self.index = []
        self.cache = {}
        self.meta_cache = {}
        self.bad_meta = []
        for ep_id, ep_dir in enumerate(self.episodes):
            traj_file = ep_dir / "trajectory.npz"
            if not traj_file.exists():
                continue
            try:
                data = np.load(traj_file)
            except Exception as e:
                if verbose:
                    print(f"[WARN] Skipping corrupt trajectory {traj_file}: {e}", file=sys.stderr)
                continue
            T = data["obs_state"].shape[0]
            if T < 2:
                continue
            for t in range(T - 1):
                self.index.append((ep_id, t))
        if verbose:
            print(f"[INFO] Dataset built: episodes={len(self.episodes)} samples={len(self.index)} bad_meta={len(self.bad_meta)}")

    def __len__(self):
        return len(self.index)

    def _load_ep(self, ep_id):
        if ep_id not in self.cache:
            self.cache[ep_id] = np.load(self.episodes[ep_id] / "trajectory.npz")
        return self.cache[ep_id]

    def _load_meta(self, ep_id):
        if ep_id in self.meta_cache:
            return self.meta_cache[ep_id]
        meta_path = self.episodes[ep_id] / "meta.json"
        meta = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            except Exception as e:
                self.bad_meta.append(str(meta_path))
                meta = {}
        self.meta_cache[ep_id] = meta
        return meta

    def _select_instruction(self, meta):
        base = meta.get("instruction_base") or meta.get("instruction") or "pick and place the red cube onto the green target"
        if not self.use_paraphrase:
            return base
        ph = meta.get("instruction_paraphrases", [])
        if ph:
            return self.rng.choice([base] + ph)
        return base

    def __getitem__(self, idx):
        ep_id, t = self.index[idx]
        ep = self._load_ep(ep_id)
        meta = self._load_meta(ep_id)
        instruction = self._select_instruction(meta)

        state = ep["obs_state"][t]
        next_state = ep["obs_state"][t+1]
        phase_id = label_phase(state)

        return {
            "obs_rgb": ep["obs_rgb"][t],
            "next_obs_rgb": ep["obs_rgb"][t+1],
            "obs_state": state,
            "next_obs_state": next_state,
            "action": ep["actions"][t],
            "done": ep["dones"][t],
            "success": bool(ep["success"][0]),
            "phase_id": phase_id,
            "instruction": instruction,
            "subgoal": meta.get("type","BASE")
        }

if __name__ == "__main__":
    ds = MjPickPlaceOfflineDataset("data/raw/mj_pick_place_v5", use_paraphrase=False)
    print("size", len(ds))
    if len(ds) > 0:
        print("sample keys:", ds[0].keys())
    if ds.bad_meta:
        print("Bad meta files:")
        for p in ds.bad_meta[:10]:
            print(" -", p)
