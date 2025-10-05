"""
Placeholder for converting raw collected trajectories into a LeRobot-style dataset object.

Later Steps:
1. Wrap each trajectory into a standardized sample dict.
2. Provide __len__, __getitem__.
3. Add transforms (normalization, cropping).
4. Integrate instruction metadata (future).
"""
import os
import numpy as np
from pathlib import Path

class PickPlaceOfflineDataset:
    def __init__(self, root: str):
        self.root = Path(root)
        self.episode_dirs = sorted([d for d in self.root.glob("episode_*") if d.is_dir()])
        self.index = []
        for ep_id, ep_dir in enumerate(self.episode_dirs):
            data = np.load(ep_dir / "trajectory.npz")
            T = data["obs_state"].shape[0]
            # We store (episode, t) indices except last frame if pairing with action.
            for t in range(T - 1):
                self.index.append((ep_id, t))
        self._cache = {}

    def __len__(self):
        return len(self.index)

    def _load_episode(self, ep_id):
        if ep_id not in self._cache:
            data = np.load(self.episode_dirs[ep_id] / "trajectory.npz")
            self._cache[ep_id] = data
        return self._cache[ep_id]

    def __getitem__(self, idx):
        ep_id, t = self.index[idx]
        ep = self._load_episode(ep_id)
        item = {
            "obs_rgb": ep["obs_rgb"][t],
            "next_obs_rgb": ep["obs_rgb"][t+1],
            "obs_state": ep["obs_state"][t],
            "next_obs_state": ep["obs_state"][t+1],
            "action": ep["actions"][t],
            "done": ep["dones"][t],
            "success": bool(ep["success"][0]),
            # Future placeholders:
            "instruction": "PLACEHOLDER_INSTRUCTION",
            "subgoal": "PLACEHOLDER_SUBGOAL"
        }
        return item

if __name__ == "__main__":
    ds = PickPlaceOfflineDataset("data/raw/pick_place_v0")
    print("Dataset size:", len(ds))
    sample = ds[0]
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            print(k, v.shape, v.dtype)
        else:
            print(k, v)