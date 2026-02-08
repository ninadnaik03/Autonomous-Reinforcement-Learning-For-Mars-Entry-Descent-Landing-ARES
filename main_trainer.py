import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mars_edl_env import MarsDeepSpaceEnv

# =====================================================
# CONFIGURATION
# =====================================================
MODEL_PATH = "inator_edl_staged_landing_v2"
STATS_NAME = "vec_normalize_edl_staged_v2.pkl"

TOTAL_STEPS = 500_000   # slightly higher due to extra phase

# =====================================================
# ENVIRONMENT
# =====================================================
raw_env = DummyVecEnv([lambda: MarsDeepSpaceEnv()])

env = VecNormalize(
    raw_env,
    norm_obs=True,
    norm_reward=True,
    clip_obs=15.0,
    clip_reward=20.0
)

env.training = True
env.norm_reward = True

# =====================================================
# PPO MODEL (STAGED EDL OPTIMIZED)
# =====================================================
print("🆕 Creating PPO model (staged-landing v2)")

model = PPO(
    policy="MlpPolicy",
    env=env,

    # LR decay: aggressive early, gentle near landing
    learning_rate=lambda progress: max(5e-6, 1.5e-5 * progress),

    n_steps=6144,
    batch_size=512,
    n_epochs=10,

    gamma=0.99,
    gae_lambda=0.95,

    clip_range=0.2,

    # ↓ Lower entropy = less oscillation in chute & landing
    ent_coef=0.0007,

    vf_coef=0.25,
    max_grad_norm=0.4,

    use_sde=False,

    verbose=1,
    device="auto",
)

print("✅ PPO initialized (staged-landing v2)")

# =====================================================
# TRAIN
# =====================================================
if __name__ == "__main__":
    print("\n🚀 STARTING STAGED MARS EDL TRAINING (v2) 🚀")
    print("- Reefed chute")
    print("- Earlier braking")
    print("- Powered descent from 20 km")
    print("- Soft touchdown")

    model.learn(
        total_timesteps=TOTAL_STEPS,
        reset_num_timesteps=True,
        progress_bar=True
    )

    model.save(MODEL_PATH)
    env.save(STATS_NAME)

    print("\n✅ TRAINING COMPLETE — MODEL + STATS SAVED")
