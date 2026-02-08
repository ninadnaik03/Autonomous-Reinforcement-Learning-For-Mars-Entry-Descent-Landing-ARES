import numpy as np
from stable_baselines3 import PPO
from mars_edl_env import MarsDustStormEnv

# Load the POLISHED final model
model = PPO.load("mars_edl_ppo_final")
env = MarsDustStormEnv()

stats = {"Success": 0, "Crash": 0, "Fly-Away": 0, "Off-Pad": 0}
total_runs = 500

print(f"Executing 500 Stress Test Missions on 'mars_edl_ppo_final'...")

for i in range(total_runs):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    
    # 1. Fly-Away Check (Still in air when time ran out)
    if env.h > 50.0: 
        stats["Fly-Away"] += 1
    # 2. Successful Landing Check
    elif abs(env.v) < 4.0 and abs(env.vx) < 2.5:
        if abs(env.x) < 25.0:
            stats["Success"] += 1
        else:
            stats["Off-Pad"] += 1
    # 3. Impact/Crash
    else:
        stats["Crash"] += 1

print("\n" + "="*30)
print("   FINAL MISSION REPORT")
print("="*30)
for key in ["Success", "Off-Pad", "Crash", "Fly-Away"]:
    print(f"{key:10}: {stats[key]:3} ({stats[key]/total_runs*100:5.1f}%)")
print("="*30)