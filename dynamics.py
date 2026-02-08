import numpy as np
from constants import *
from atmosphere import mars_density

def edl_dynamics_2d(
    t,
    state,
    thrust,
    theta,
    mass,
    wind_speed,
    area
):
    x, h, vx, v, pitch, pitch_rate = state

    rho = mars_density(h)

    v_rel_x = vx - wind_speed
    v_total = np.sqrt(v_rel_x**2 + v**2) + 1e-6

    # ===============================
    # AERODYNAMICS
    # ===============================
    q = 0.5 * rho * v_total**2

    # Drag
    Cd = CD_CHUTE if area >= 0.5 * CHUTE_AREA else CD_BODY
    drag = q * Cd * area

    # 🔥 STRONG LIFT (ENTRY AEROSHELL)
    if area > REFERENCE_AREA:
        Cl = 1.2    # deliberately strong for Mars EDL
    else:
        Cl = 0.2

    lift = q * Cl * area

    # Unit vectors
    vx_hat = v_rel_x / v_total
    vy_hat = v / v_total

    # Forces
    drag_x = -drag * vx_hat
    drag_y = -drag * vy_hat

    # 🔥 Lift perpendicular to velocity (NO attenuation)
    lift_x = -lift * vy_hat
    lift_y =  lift * vx_hat

    # Parachute kills horizontal
    if area >= 0.5 * CHUTE_AREA:
        drag_x *= 1.8

    # Thrust
    thrust_x = thrust * np.sin(theta)
    thrust_y = thrust * np.cos(theta)

    # Accelerations
    ax = (drag_x + lift_x + thrust_x) / mass
    ay = (drag_y + lift_y + thrust_y) / mass - MARS_GRAVITY

    # Attitude dynamics
    dpitch_rate = -2.5 * pitch_rate - 4.0 * pitch + theta

    # Fuel
    m_dot = thrust / (ISP * 9.80665) if thrust > 0 else 0.0

    # Derivatives
    dx = vx
    dh = v
    dvx = ax
    dv  = ay

    return dx, dh, dvx, dv, dpitch_rate, m_dot
