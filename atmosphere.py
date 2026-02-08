import numpy as np
from constants import MARS_RHO0, MARS_SCALE_HEIGHT

def mars_density(altitude):
    """
    Exponential Mars atmosphere model.
    Valid from surface up to ~130 km.
    """
    altitude = np.clip(altitude, 0.0, 130000.0)
    return MARS_RHO0 * np.exp(-altitude / MARS_SCALE_HEIGHT)
