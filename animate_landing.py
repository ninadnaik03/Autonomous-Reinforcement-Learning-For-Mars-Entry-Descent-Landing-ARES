import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from mars_edl_env import MarsDeepSpaceEnv
from atmosphere import mars_density
from constants import FUEL_MASS, MAX_THRUST, MARS_GRAVITY, DT_PHYSICS

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = "inator_edl_staged_landing_v2"
STATS_PATH = "vec_normalize_edl_staged_v2.pkl"
TOUCHDOWN_ALT = 0.5

ALTITUDE_BINS = [70000, 60000, 50000, 40000, 30000, 20000, 10000, 5000, 1000, 0]
altitude_log = {h: None for h in ALTITUDE_BINS}

# =====================================================
# ENV
# =====================================================
raw_env = DummyVecEnv([lambda: MarsDeepSpaceEnv()])
base_env = raw_env.envs[0]

if os.path.exists(STATS_PATH):
    env = VecNormalize.load(STATS_PATH, raw_env)
    env.training = False
    env.norm_reward = False
else:
    env = raw_env

model = PPO.load(MODEL_PATH, env=env)
obs = env.reset()

# =====================================================
# STATE
# =====================================================
landed = False
touchdown = {}
xs, hs = [], []
prev_v = None
peak_heat = 0.0

# =====================================================
# FIGURE
# =====================================================
fig, ax = plt.subplots(figsize=(13, 12))
ax.set_xlim(-40000, 40000)
ax.set_ylim(-2000, 120000)
ax.set_xlabel("Downrange (m)")
ax.set_ylabel("Altitude (m)")
ax.set_title("Mars EDL — Flight Director Status View")
ax.axhline(0, color="brown", lw=3)

traj, = ax.plot([], [], "b-", lw=1.6)
lander = ax.scatter([], [], s=120, c="red", marker="v", zorder=10)

# =====================================================
# LEFT PANEL (SPACED PROPERLY)
# =====================================================
panel_x = 0.02
panel_y = 0.95
panel_gap = 0.085   # 👈 spacing FIXED

def panel(y):
    return ax.text(
        panel_x, y, "",
        transform=ax.transAxes,
        va="top",
        family="monospace",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.96)
    )

nav_box    = panel(panel_y - panel_gap*0)
prop_box   = panel(panel_y - panel_gap*1)
att_box    = panel(panel_y - panel_gap*2)
dyn_box    = panel(panel_y - panel_gap*3)
therm_box  = panel(panel_y - panel_gap*4)
guid_box   = panel(panel_y - panel_gap*5)
status_box = panel(panel_y - panel_gap*6)

# =====================================================
# RIGHT PANEL — ALTITUDE SPEED CHECKER (RESTORED)
# =====================================================
alt_table = ax.text(
    0.72, 0.95,
    "",
    transform=ax.transAxes,
    fontsize=9,
    family="monospace",
    va="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.96)
)

# =====================================================
# PLAYBACK SPEED
# =====================================================
speed_mult = 1.0
def on_key(event):
    global speed_mult
    speed_mult = {"1":1, "2":1.5, "3":2, "4":3}.get(event.key, speed_mult)
fig.canvas.mpl_connect("key_press_event", on_key)

# =====================================================
# PHASE
# =====================================================
def phase(h):
    if h > 60000: return "ENTRY"
    if base_env.chute_deployed: return "PARACHUTE"
    if h < 2000: return "POWERED DESCENT"
    return "GUIDANCE"

# =====================================================
# UPDATE
# =====================================================
def update(_):
    global obs, landed, prev_v, peak_heat

    if not landed:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)

        if base_env.h <= TOUCHDOWN_ALT:
            landed = True
            touchdown.update(
                x=base_env.x,
                h=0.0,
                v=base_env.v,
                vx=base_env.vx,
                fuel=base_env.fuel,
                time=base_env.steps * DT_PHYSICS,
                pitch=base_env.pitch
            )
            xs.append(touchdown["x"])
            hs.append(touchdown["h"])
        else:
            xs.append(base_env.x)
            hs.append(base_env.h)

    # Use frozen values after landing
    if landed:
        x, h = touchdown["x"], touchdown["h"]
        v, vx = touchdown["v"], touchdown["vx"]
        fuel = touchdown["fuel"]
        sim_time = touchdown["time"]
        pitch = touchdown["pitch"]
    else:
        x, h = base_env.x, base_env.h
        v, vx = base_env.v, base_env.vx
        fuel = base_env.fuel
        sim_time = base_env.steps * DT_PHYSICS
        pitch = base_env.pitch

    traj.set_data(xs, hs)
    lander.set_offsets([[x, h]])

    # Heat flux
    rho = mars_density(h)
    v_tot = np.sqrt(v*v + vx*vx)
    heat_flux = 1e-4 * np.sqrt(rho) * v_tot**3
    peak_heat = max(peak_heat, heat_flux)

    # G-load
    g_load = 0.0
    if prev_v is not None and not landed:
        acc = (v - prev_v) / DT_PHYSICS
        g_load = abs(acc + MARS_GRAVITY) / 9.81
    prev_v = v

    thrust = getattr(base_env, "last_thrust", 0.0)
    thrust_pct = 100 * thrust / MAX_THRUST

    # LEFT PANEL TEXT
    nav_box.set_text(
        f"NAVIGATION\nALT : {h:8.1f} m\nVY  : {v:8.2f} m/s\nVX  : {vx:8.2f} m/s\nRNG : {x:8.1f} m"
    )

    prop_box.set_text(
        f"PROPULSION\nTHRUST : {thrust_pct:6.1f} %\nFUEL   : {100*fuel/FUEL_MASS:6.1f} %"
    )

    att_box.set_text(
        f"ATTITUDE\nPITCH : {np.degrees(pitch):6.1f} deg"
    )

    dyn_box.set_text(
        f"DYNAMICS\nG-LOAD : {g_load:6.2f} g\nPHASE  : {phase(h)}"
    )

    therm_box.set_text(
        f"THERMAL\nHEAT : {heat_flux:7.1f} kW/m²\nPEAK : {peak_heat:7.1f} kW/m²"
    )

    guid_box.set_text(
        f"GUIDANCE\nT+ : {sim_time:7.1f} s\nT-LAND : {'—' if landed else f'{h/max(abs(v),1e-3):6.1f} s'}"
    )

    status_box.set_text(
        f"STATUS\nTOUCHDOWN : {'YES' if landed else 'NO'}\nPLAYBACK : {speed_mult:.1f}×"
    )

    # ALTITUDE SPEED TABLE (RESTORED)
    for h_bin in ALTITUDE_BINS:
        if altitude_log[h_bin] is None and h <= h_bin:
            altitude_log[h_bin] = {
                "vy": v,
                "vx": vx,
                "vt": np.sqrt(v*v + vx*vx)
            }

    lines = ["ALT(km) |   VY   |   VX   |   VT",
             "--------------------------------"]
    for h_bin in ALTITUDE_BINS:
        e = altitude_log[h_bin]
        if e is None:
            lines.append(f"{h_bin//1000:>6} |   ---  |   ---  |   ---")
        else:
            lines.append(
                f"{h_bin//1000:>6} | {e['vy']:6.1f} | {e['vx']:6.1f} | {e['vt']:6.1f}"
            )

    alt_table.set_text("\n".join(lines))

    return (
        traj, lander,
        nav_box, prop_box, att_box, dyn_box,
        therm_box, guid_box, status_box, alt_table
    )

# =====================================================
# RUN
# =====================================================
anim = FuncAnimation(fig, update, interval=20, blit=False)
plt.tight_layout()
plt.show()
