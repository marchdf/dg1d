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
def advection_upwinding(ul,ur):
    """Returns the interface flux for the advection equation (simple upwinding)"""
    return ul


#================================================================================
def euler_rusanov(ul,ur):
    """Returns the Rusanov interface flux for the Euler equations

    V. V. Rusanov, Calculation of Interaction of Non-Steady Shock Waves with Obstacles, J. Comput. Math. Phys. USSR, 1, pp. 267-279, 1961.

    """

    gamma = 1.4
    F = np.zeros(ul.shape)
        
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
