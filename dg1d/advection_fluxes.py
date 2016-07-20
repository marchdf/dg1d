#================================================================================
#
# Imports
#
#================================================================================
import sys
import numpy as np

#================================================================================
#
# Function definitions
#
#================================================================================

#================================================================================
def max_wave_speed(u):
    """Returns the maximum wave speed for advection"""
    return 1

#================================================================================
def riemann_upwinding(ul,ur):
    """Returns the interface flux for the advection equation (simple upwinding)"""
    return ul

#================================================================================
def interior_flux(ug):
    """Returns the interior flux for the advection equation"""
    return ug

