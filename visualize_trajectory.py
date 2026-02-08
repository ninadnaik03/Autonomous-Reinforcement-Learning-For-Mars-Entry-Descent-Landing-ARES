import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from mars_edl_env import MarsDeepSpaceEnv
import os

def run_telemetry_probe():
    stats_path = "vec_normalize_aero.pkl"
    model_path = "inator_aero_master_v1.zip"

    if not os.path.exists(model_path):
        print("Model file not found. Ensure the training script has saved at least once.")
        return

    raw_env = DummyVecEnv([lambda: MarsDeepSpaceEnv()])
    env = VecNormalize.load(stats_path, raw_env)
    env.training = False
    env.norm_reward = False
    
    model = PPO.load(model_path, env=env)

    print(f"{'='*60}")
    print(f"{'TEST FLIGHT TELEMETRY REPORT':^60}")
    print(f"{'='*60}")

    for i in range(5):
        obs = env.reset()
        done = False
        
        chute_alt = None
        max_v = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Use raw env to capture exact parameters
            inner = raw_env.envs[0]
            if inner.chute_deployed and chute_alt is None:
                chute_alt = inner.h
            max_v = max(max_v, abs(inner.v))
            
        inner = raw_env.envs[0]
        mach_entry = max_v / 240.0 # Approx speed of sound on Mars
        
        print(f"FLIGHT #{i+1}")
        print(f"  - Status:      {'[SUCCESS]' if inner.h <= 5 and abs(inner.v) < 5 else '[CRASH]'}")
        print(f"  - Entry Speed: Mach {mach_entry:.2f} ({max_v:.1f} m/s)")
        print(f"  - Chute Deployed at: {f'{chute_alt:.1f} m' if chute_alt else 'NOT DEPLOYED'}")
        print(f"  - Final V-Speed:     {inner.v:.2f} m/s")
        print(f"  - Fuel Remaining:    {inner.fuel:.1f} kg")
        print(f"-"*30)

if __name__ == "__main__":
    run_telemetry_probe()