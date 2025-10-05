import numpy as np
from envs.mj_pick_place_env import MjPickPlaceEnv, MjPickPlaceConfig

def main():
    env = MjPickPlaceEnv(MjPickPlaceConfig(debug=True,lift_only=False))
    obs = env.reset()
    for t in range(100):
        s=obs["state"]; eef=s[0:3]; cube=s[4:7]; tgt=s[7:9]
        action = np.array([
            0.0,
            0.0,
            -0.3 if t<15 else (0.3 if 15<=t<35 else 0.0),
            1.0 if t>12 else 0.0
        ], dtype=np.float32)
        obs,r,d,info = env.step(action)
        if t%10==0:
            dist = np.linalg.norm(cube[:2]-tgt)
            print(f"t={t} eef={eef.round(3)} cube={cube.round(3)} dist={dist:.3f} attached={info['success']}")
        if d:
            print("Done", info)
            break
    env.close()

if __name__=="__main__":
    main()
