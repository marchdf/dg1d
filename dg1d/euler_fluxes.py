#================================================================================
#
# Imports
#
#================================================================================
import sys
import numpy as np

#================================================================================
#
# Constants
#
#================================================================================
gamma = 1.4


#================================================================================
#
# Function definitions
#
#================================================================================

#================================================================================
def max_wave_speed(u):
    """Returns the maximum wave speed for the Euler system"""

    # Initialize
    maxvap = 0 

    # Primitive variables
    rho = u[0,0::3]
    v   = u[0,1::3]/rho
    E   = u[0,2::3]
    p   = (gamma-1)*(E-0.5*rho*v*v)    

    # Get the wave speed
    wave_speed = np.fabs(v) + np.sqrt(gamma*p/rho)

    return np.max(wave_speed)

#================================================================================
def riemann_rusanov(ul,ur):
    """Returns the Rusanov interface flux for the Euler equations

    V. V. Rusanov, Calculation of Interaction of Non-Steady Shock Waves with Obstacles, J. Comput. Math. Phys. USSR, 1, pp. 267-279, 1961.

    """

    # Initialize
    F = np.zeros(ul.shape)

    # Primitive variables and sound speeds
    rhoL = ul[0::3]
    vL   = ul[1::3]/rhoL
    EL   = ul[2::3]
    pL   = (gamma-1)*(EL-0.5*rhoL*vL*vL)    
    aL   = np.sqrt(gamma*pL/rhoL)

    rhoR = ur[0::3]
    vR   = ur[1::3]/rhoR
    ER   = ur[2::3]
    pR   = (gamma-1)*(ER - 0.5*rhoR*vR*vR)
    aR   = np.sqrt(gamma*pR/rhoR)

    # Find the maximum eigenvalue for each interface
    maxvap = np.maximum(np.fabs(vL)+aL,np.fabs(vR)+aR)

    # first: fx = rho*u
    F[0::3] = 0.5*(rhoL*vL + rhoR*vR - maxvap*(rhoR-rhoL))
    
    # second: fx = rho*u*u+p
    F[1::3] = 0.5*(rhoL*vL*vL+pL  + rhoR*vR*vR+pR - maxvap*(rhoR*vR-rhoL*vL))
    
    # third: fx = (E+p)*u
    F[2::3] = 0.5*((EL+pL)*vL + (ER+pR)*vR - maxvap*(ER-EL))

    return F


#================================================================================
def interior_flux(ug):
    """Returns the interior flux for the Euler equations"""

    # Initialize
    F = np.zeros(ug.shape)

    # Primitive variables
    rho = ug[:,0::3]
    v   = ug[:,1::3]/rho
    E   = ug[:,2::3]
    p   = (gamma-1)*(E-0.5*rho*v*v)    
    
    # Flux in x-direction
    F[:,0::3] = ug[:,1::3]
    F[:,1::3] = rho*v*v + p
    F[:,2::3] = (E+p)*v
    
    return F
