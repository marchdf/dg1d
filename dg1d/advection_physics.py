# ========================================================================
#
# Imports
#
# ========================================================================
import numpy as np

# ========================================================================
#
# Function definitions
#
# ========================================================================

# ========================================================================


def max_wave_speed(u):
    """Returns the maximum wave speed for advection"""
    return 1

# ========================================================================


def riemann_upwinding(ul, ur):
    """Returns the interface flux for the advection equation (simple upwinding)"""
    return ul

# ========================================================================


def interior_flux(ug):
    """Returns the interior flux for the advection equation"""
    return ug

# ========================================================================


def sensing(sensors, thresholds, solution):
    """A simple sensor which just calculates the difference between the
       left/right cell solutions for the advection equation.
    """

    # left/right solution
    ul = solution.u[0, :-solution.N_F]
    ur = solution.u[0, solution.N_F:]

    # Calculate the sensor
    phi = np.fabs(ur - ul)
    PHI = 2 * phi / ((1 + phi) * (1 + phi))

    # Find where the sensor exceeds the threshold value
    idx = np.array(np.where(PHI > thresholds[0]))
    sensors[idx] = 1
    sensors[idx + 1] = 1
