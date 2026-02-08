import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from mars_edl_env import MarsDeepSpaceEnv # Use the Orbital Env

# --- 1. Load the Legend & Stats ---
raw_env = DummyVecEnv([lambda: MarsDeepSpaceEnv()])
stats_path = "vec_normalize_final.pkl"
model_path = "inator_orbital_final_legend"

if os.path.exists(stats_path):
    # Load with training=False so it doesn't keep updating stats during the test
    env = VecNormalize.load(stats_path, raw_env)
    env.training = False
    env.norm_reward = False
    print(f"Stats loaded from {stats_path}")
else:
    print("Warning: .pkl not found. Telemetry might be skewed.")
    env = raw_env

model = PPO.load(model_path)

# --- 2. Reporting Header ---
print(f"\n{'Run':<4} | {'Max Mach':<8} | {'Brake Alt':<10} | {'V-Touch':<8} | {'Fuel %':<7} | {'Result'}")
print("-" * 65)

for i in range(1, 11):
    obs = env.reset()
    done = False
    actual_env = raw_env.envs[0]
    
    max_mach = 0
    brake_altitude = None
    initial_fuel = actual_env.fuel

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_flag, info = env.step(action)
        
        # Track Mach (Speed of Sound on Mars ~ 240 m/s)
        current_mach = np.sqrt(actual_env.v**2 + actual_env.vx**2) / 240.0
        if current_mach > max_mach: max_mach = current_mach
        
        # Detect first engine ignition (The "Big Brake")
        if brake_altitude is None and actual_env.fuel < (initial_fuel - 5.0):
            brake_altitude = actual_env.h

        if isinstance(done_flag, (list, np.ndarray)): done_flag = done_flag[0]
        if done_flag: done = True

    # --- 3. Score the Result ---
    # Success = Vertical speed < 3.5 m/s and landed safely
    fuel_left_pc = (actual_env.fuel / initial_fuel) * 100
    result = "LEGEND" if (abs(actual_env.v) < 3.5 and actual_env.h <= 10) else "CRASH"
    
    # Format brake altitude for display
    b_alt_str = f"{brake_altitude/1000:>6.1f}km" if brake_altitude else "N/A"

    print(f"{i:<4} | {max_mach:>7.1f} | {b_alt_str:<10} | {actual_env.v:>6.1f}m/s | {fuel_left_pc:>5.1f}% | {result}")