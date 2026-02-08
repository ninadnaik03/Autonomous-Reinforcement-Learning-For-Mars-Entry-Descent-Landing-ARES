import numpy as np

# ===============================
# PHYSICS
# ===============================
MARS_GRAVITY = 3.711
DT_PHYSICS = 0.1

# ===============================
# ATMOSPHERE
# ===============================
MARS_RHO0 = 0.020
MARS_SCALE_HEIGHT = 11100.0

# ===============================
# AERODYNAMICS
# ===============================
REFERENCE_AREA = 15.0
CHUTE_AREA = 320.0

CD_BODY = 1.3
CD_CHUTE = 1.75

Q_MAX_CHUTE = 12000.0

# ===============================
# PROPULSION
# ===============================
MAX_THRUST = 120000.0
ISP = 330.0

# ===============================
# MASS
# ===============================
DRY_MASS = 1200.0
FUEL_MASS = 9000.0
INITIAL_MASS = DRY_MASS + FUEL_MASS

# ===============================
# LANDING LIMITS
# ===============================
SAFE_LANDING_VEL = 6.0
SAFE_LANDING_VX  = 2.0

# ===============================
# NUMERICAL GUARDS
# ===============================
MAX_ALTITUDE = 130000.0
MIN_ALTITUDE = -500.0
