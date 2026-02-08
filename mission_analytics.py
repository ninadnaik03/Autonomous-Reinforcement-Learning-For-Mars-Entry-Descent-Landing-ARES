import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mars_edl_env import MarsDeepSpaceEnv
from constants import (
    SAFE_LANDING_VEL,
    SAFE_LANDING_VX,
    FUEL_MASS,
    DRY_MASS,
    MAX_THRUST,
    MARS_GRAVITY,
)

# =====================================================
# CONFIG (CANONICAL FILES)
# =====================================================
MODEL_PATH = "inator_edl_fresh_run_v1"
STATS_PATH = "vec_normalize_edl_fresh_v1.pkl"

# =====================================================
# LOAD ENV
# =====================================================
raw_env = DummyVecEnv([lambda: MarsDeepSpaceEnv()])

if not os.path.exists(STATS_PATH):
    raise RuntimeError("VecNormalize stats not found. Train first.")

env = VecNormalize.load(STATS_PATH, raw_env)
env.training = False
env.norm_reward = False

base_env = raw_env.envs[0]

# =====================================================
# LOAD MODEL
# =====================================================
model = PPO.load(MODEL_PATH, env=env)

# =====================================================
# RESET
# =====================================================
obs = env.reset()
done = False
steps = 0

# =====================================================
# PHASE DETECTION
# =====================================================
def detect_phase():
    mass = DRY_MASS + base_env.fuel
    a_max = MAX_THRUST / mass - MARS_GRAVITY
    a_max = max(a_max, 0.1)
    h_stop = (base_env.v ** 2) / (2 * a_max) if base_env.v < 0 else 0.0

    if base_env.h > 60000:
        return "ENTRY"
    elif base_env.chute_deployed:
        return "PARACHUTE"
    elif base_env.h < 1.3 * h_stop:
        return "POWERED_DESCENT"
    else:
        return "GUIDANCE"

last_phase = detect_phase()

# =====================================================
# ROLLOUT (VECENV SAFE)
# =====================================================
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    steps += 1
    last_phase = detect_phase()

# =====================================================
# RESULTS
# =====================================================
v_vert = base_env.v
v_horz = base_env.vx
v_total = np.sqrt(v_vert**2 + v_horz**2)
fuel_pct = 100 * base_env.fuel / FUEL_MASS

landed_soft = (
    base_env.h <= 0.0
    and abs(v_vert) < SAFE_LANDING_VEL
    and abs(v_horz) < SAFE_LANDING_VX
)

print("\n================ EDL RESULT ================")
print(f"Steps simulated        : {steps}")
print(f"Final altitude         : {base_env.h:.2f} m")
print(f"Vertical speed         : {v_vert:.2f} m/s")
print(f"Horizontal speed       : {v_horz:.2f} m/s")
print(f"Total speed            : {v_total:.2f} m/s")
print(f"Fuel remaining         : {fuel_pct:.2f} %")
print(f"Parachute deployed     : {bool(base_env.chute_deployed)}")
print(f"Last phase             : {last_phase}")

if landed_soft:
    print("✅ LANDING STATUS: SOFT LANDING")
else:
    if base_env.h <= 0:
        print("❌ LANDING STATUS: CRASHED")
    else:
        print("❌ LANDING STATUS: TERMINATED EARLY")

print("============================================\n")
