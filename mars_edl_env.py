import gymnasium as gym
from gymnasium import spaces
import numpy as np

from dynamics import edl_dynamics_2d
from constants import *

ENTRY_AEROSHELL_AREA = 8.0 * REFERENCE_AREA
REEF_CHUTE_AREA = 0.6 * CHUTE_AREA
FULL_CHUTE_AREA = CHUTE_AREA


class MarsDeepSpaceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(8,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.dt = DT_PHYSICS

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.h  = np.random.uniform(115_000.0, 120_000.0)
        self.v  = np.random.uniform(-1900.0, -1700.0)
        self.x  = np.random.uniform(-3000.0, 3000.0)
        self.vx = np.random.uniform(-50.0, 50.0)

        self.pitch = 0.0
        self.pitch_rate = 0.0

        self.fuel = FUEL_MASS
        self.wind = np.random.uniform(-20.0, 20.0)

        self.chute = 0.0
        self.chute_deployed = 0.0

        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.h / 120000.0,
            self.v / 2000.0,
            self.vx / 800.0,
            self.pitch,
            self.pitch_rate,
            self.fuel / FUEL_MASS,
            self.wind / 60.0,
            self.chute
        ], dtype=np.float32)

    def step(self, action):
        prev_v  = self.v
        prev_vx = self.vx

        thrust_cmd = np.clip((action[0] + 1.0) / 2.0, 0.0, 1.0)
        tilt_cmd = np.clip(action[1], -1.0, 1.0)

        # ==================================================
        # GUIDANCE
        # ==================================================

        # ---------- ENTRY (> 60 km)
        if self.h > 60000.0:
            self.chute = 0.0
            thrust = 0.0
            tilt = np.radians(70.0) * np.sign(self.vx + 1e-3)
            target_v = -1300.0
            area = ENTRY_AEROSHELL_AREA

        # ---------- REEFED CHUTE (60 → 40 km)
        elif self.h > 40000.0:
            self.chute = 0.5
            thrust = 0.0
            tilt = tilt_cmd * np.radians(15.0)
            target_v = -900.0
            area = REEF_CHUTE_AREA

        # ---------- FULL CHUTE (40 → 20 km)
        elif self.h > 20000.0:
            self.chute = 1.0
            thrust = 0.0
            tilt = tilt_cmd * np.radians(8.0)
            target_v = -600.0
            area = FULL_CHUTE_AREA

        # ==================================================
        # 🔥 POWERED DESCENT (< 20 km) — ANTI-HOVER FIX
        # ==================================================
        else:
            self.chute = 0.0
            target_v = -2.0

            mass = DRY_MASS + self.fuel

            # ---------------------------
            # VERTICAL CONTROL (CRITICAL)
            # ---------------------------
            if self.v < -1.5:
                # Falling too fast → brake
                thrust = mass * (MARS_GRAVITY + 0.8 * abs(self.v))
            else:
                # Near zero or upward → FORCE descent
                thrust = mass * (MARS_GRAVITY - 0.6)

            thrust = np.clip(thrust, 0.0, MAX_THRUST)

            # ---------------------------
            # HORIZONTAL KILL
            # ---------------------------
            if self.h > 500.0:
                tilt = np.clip(-0.4 * self.vx, -1.0, 1.0) * np.radians(8.0)
            else:
                tilt = np.clip(-1.2 * self.vx, -1.0, 1.0) * np.radians(15.0)

            area = REFERENCE_AREA

            # Absolute no ascent
            if self.v > 0.0:
                thrust = min(thrust, 0.7 * mass * MARS_GRAVITY)

            # Kill skating
            if self.h < 50.0:
                self.vx *= 0.5

        self.chute_deployed = self.chute
        mass = DRY_MASS + self.fuel

        dx, dh, dvx, dv, dpitch, m_dot = edl_dynamics_2d(
            0.0,
            [self.x, self.h, self.vx, self.v, self.pitch, self.pitch_rate],
            thrust,
            tilt,
            mass,
            self.wind,
            area
        )

        self.x  += dx * self.dt
        self.h  += dh * self.dt
        self.vx += dvx * self.dt
        self.v  += dv * self.dt

        self.pitch_rate += dpitch * self.dt
        self.pitch += self.pitch_rate * self.dt

        # Final ground capture
        if self.h < 5.0:
            self.v = min(self.v, -0.5)

        self.fuel = max(0.0, self.fuel - m_dot * self.dt)
        self.steps += 1

        # ==================================================
        # REWARD
        # ==================================================
        reward = -0.05
        reward -= 0.1 * abs(self.v - target_v)
        reward -= 0.02 * self.v**2
        reward -= 0.02 * self.vx**2

        if self.h > 60000.0:
            reward += 0.05 * abs(self.vx)

        if self.chute > 0.0:
            reward -= 0.25 * abs(self.vx)

        done = False
        if self.h <= 0.0:
            done = True
            if abs(prev_v) < SAFE_LANDING_VEL and abs(prev_vx) < SAFE_LANDING_VX:
                reward += 20000.0
            else:
                reward -= 15000.0

        if self.h < 200.0:
            reward -= 400.0 * abs(self.v + 2.0)
            reward -= 600.0 * abs(self.vx)

        if self.v > 0.0:
            reward -= 5000.0 * self.v

        truncated = self.steps > 40000
        terminated = self.h > MAX_ALTITUDE or abs(self.x) > 150000.0

        return self._get_obs(), float(reward), done or terminated, truncated, {}
