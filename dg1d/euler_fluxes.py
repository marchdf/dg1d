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

    # Loop on all the elements
    for i in range(0,len(ul),3):

        # Left side
        rhoL = ul[i]
        vL   = ul[i+1]/rhoL
        EtL  = ul[i+2]

        # Right side
        rhoR = ur[i]
        vR   = ur[i+1]/rhoR
        EtR  = ur[i+2]
        
        # Pressure
        pL = (gamma-1)*(EtL - 0.5*rhoL*vL*vL)
        pR = (gamma-1)*(EtR - 0.5*rhoR*vR*vR)
        
        # Sound speed
        aL = np.sqrt(gamma*pL/rhoL)
        aR = np.sqrt(gamma*pR/rhoR)
        
        # Find the maximum eigenvalue
        maxvap = np.max([np.fabs(vL)+aL, np.fabs(vR)+aR])

        # first: fx = rho*u
        F[i] = 0.5*(rhoL*vL + rhoR*vR - maxvap*(rhoR-rhoL))
        
        # second: fx = rho*u*u+p
        F[i+1] = 0.5*(rhoL*vL*vL+pL  + rhoR*vR*vR+pR - maxvap*(rhoR*vR-rhoL*vL))
        
        # third: fx = (Et+p)*u
        F[i+2] = 0.5*((EtL+pL)*vL + (EtR+pR)*vR - maxvap*(EtR-EtL))
                                                                
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
