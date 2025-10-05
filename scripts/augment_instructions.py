import argparse, json, random
from pathlib import Path
from typing import List

BASIC_SYNONYMS = [
    ("pick", ["grasp", "lift", "pick up"]),
    ("place", ["put", "set", "position"]),
    ("red cube", ["crimson block", "red block"]),
    ("green target", ["green pad", "green marker", "green zone"]),
]

BASE_SENTENCE = "pick and place the red cube onto the green target"

def heuristic_paraphrases(base: str, n: int, seed: int) -> List[str]:
    random.seed(seed)
    variants = set([base])
    while len(variants) < n + 1:
        s = base
        for root, alts in BASIC_SYNONYMS:
            if root in s and random.random() < 0.5:
                s = s.replace(root, random.choice(alts))
        variants.add(s)
    variants.discard(base)
    return list(variants)[:n]

def hf_generate(model_name: str, base: str, n: int, max_new_tokens: int = 30):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    outs = []
    for _ in range(n):
        prompt = f"Paraphrase the instruction preserving meaning:\n{base}\nParaphrase:"
        ids = tok.encode(prompt, return_tensors="pt").to(mdl.device)
        gen = mdl.generate(ids, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.9, temperature=0.8)
        text = tok.decode(gen[0], skip_special_tokens=True)
        candidate = text.split("Paraphrase:")[-1].strip()
        if candidate and candidate.lower() != base.lower():
            outs.append(candidate)
    cleaned = []
    for c in outs:
        if c not in cleaned:
            cleaned.append(c)
    return cleaned

def main(args):
    root = Path(args.data_root)
    episodes = sorted([d for d in root.glob("episode_*") if d.is_dir()])
    if not episodes:
        print("No episodes found.")
        return
    if args.hf_model:
        paraphrases = hf_generate(args.hf_model, BASE_SENTENCE, args.num)
    else:
        paraphrases = heuristic_paraphrases(BASE_SENTENCE, args.num, args.seed)
    print("Paraphrases:")
    for p in paraphrases:
        print(" -", p)
    for ep_dir in episodes:
        meta_path = ep_dir / "meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path, "r") as f:
            meta = json.load(f)
        meta["instruction_base"] = BASE_SENTENCE
        meta["instruction_paraphrases"] = paraphrases
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    print("Updated meta.json files.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/raw/mj_pick_place_v0")
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hf-model", type=str, default="")
    args = ap.parse_args()
    main(args)
