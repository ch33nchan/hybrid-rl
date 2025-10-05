# MuJoCo Pick & Place Sandbox (Phase 0)

## Setup
python3.11 -m venv .venv
source .venv/bin/activate
pip3.11 install --upgrade pip setuptools wheel
pip3.11 install -r requirements.txt

## Collect Data
python3.11 scripts/collect_mj_pick_place.py --episodes 25 --output data/raw/mj_pick_place_v0

## Inspect Dataset
python3.11 lerobot_dataset/pick_place_mj_builder.py

## Train BC
python3.11 train/train_bc.py --data_root data/raw/mj_pick_place_v0 --epochs 5

## Next
- Add instruction paraphrases
- Integrate vision encoder
- Planner latent extraction
- Diffusion policy + critic

## Updated Dataset Version (v1)
Improved scripted policy & success logic:

Collect:
python3.11 -m scripts.collect_mj_pick_place --episodes 50 --output data/raw/mj_pick_place_v1

Train:
python3.11 -m train.train_bc --data_root data/raw/mj_pick_place_v1 --epochs 5

Evaluate:
python3.11 -m scripts.eval_bc_policy --checkpoint models/ckpts/bc_policy.pt --episodes 20

## Dataset v2 (improved scripted success)
Collect:
python3.11 -m scripts.collect_mj_pick_place --episodes 100 --output data/raw/mj_pick_place_v2
Benchmark scripted (no saving):
python3.11 -m scripts.benchmark_scripted --episodes 50
Train BC on v2:
python3.11 -m train.train_bc --data_root data/raw/mj_pick_place_v2 --epochs 5
Evaluate:
python3.11 -m scripts.eval_bc_policy --checkpoint models/ckpts/bc_policy.pt --episodes 30
