#================================================================================
#
# Imports
#
#================================================================================
import sys
import numpy as np
import dg1d.constants as constants

#================================================================================
#
# Function definitions
#
#================================================================================

#================================================================================
def max_wave_speed(u):
    """Returns the maximum wave speed for the Euler system"""

    # Primitive variables
    rho = u[0,0::3]
    v   = u[0,1::3]/rho
    E   = u[0,2::3]
    p   = (constants.gamma-1)*(E-0.5*rho*v*v)    

    # Get the wave speed
    wave_speed = np.fabs(v) + np.sqrt(constants.gamma*p/rho)

    return np.max(wave_speed)

#================================================================================
def riemann_rusanov(ul,ur):
    """Returns the Rusanov interface flux for the Euler equations

    V. V. Rusanov, Calculation of Interaction of Non-Steady Shock Waves with Obstacles, J. Comput. Math. Phys. USSR, 1, pp. 267-279, 1961.

    Heavily inspired/taken from "I Do Like CFD" website: http://ossanworld.com/cfdbooks/cfdcodes/oned_euler_fluxes_v5.f90

    """

    # Initialize
    F = np.zeros(ul.shape)

    # Primitive variables and sound speeds
    rhoL = ul[0::3]
    vL   = ul[1::3]/rhoL
    EL   = ul[2::3]
    pL   = (constants.gamma-1)*(EL-0.5*rhoL*vL*vL)    
    aL   = np.sqrt(constants.gamma*pL/rhoL)

    rhoR = ur[0::3]
    vR   = ur[1::3]/rhoR
    ER   = ur[2::3]
    pR   = (constants.gamma-1)*(ER - 0.5*rhoR*vR*vR)
    aR   = np.sqrt(constants.gamma*pR/rhoR)

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
def riemann_godunov(ul,ur):
    """Returns the Godunov interface flux for the Euler equations

    S. K. Godunov, A Difference Scheme for Numerical Computation of Discontinuous Solution of Hydrodynamic Equations, Math. Sbornik, 47, pp. 271-306, 1959 (in Russian). Translated US Joint Publ. Res. Service, JPRS 7226 (1969)

    Taken from "I Do Like CFD" website: http://ossanworld.com/cfdbooks/cfdcodes/oned_euler_fluxes_v5.f90

    """

    # Initialize
    F = np.zeros(ul.shape)

    # Primitive variables and sound speeds
    rhoL = ul[0::3]
    vL   = ul[1::3]/rhoL
    EL   = ul[2::3]
    pL   = (constants.gamma-1)*(EL-0.5*rhoL*vL*vL)    
    aL   = np.sqrt(constants.gamma*pL/rhoL)
    
    rhoR = ur[0::3]
    vR   = ur[1::3]/rhoR
    ER   = ur[2::3]
    pR   = (constants.gamma-1)*(ER - 0.5*rhoR*vR*vR)
    aR   = np.sqrt(constants.gamma*pR/rhoR)

    # Fixed point iteration tolerance
    tol = 1e-6
    
    # Loop over each interface
    for i in range(len(rhoL)):
        
        # Supersonic flow to the right
        if (vL[i]/aL[i] >= 1.0):
            F[i*3+0] = rhoL[i]*vL[i]                 # first: fx = rho*u
            F[i*3+1] = rhoL[i]*vL[i]*vL[i]+pL[i]     # second: fx = rho*u*u+p
            F[i*3+2] = (EL[i]+pL[i])*vL[i]           # third: fx = (E+p)*u

        # Supersonic flow to the left
        elif (vR[i]/aR[i] <= -1.0):
            F[i*3+0] = rhoR[i]*vR[i]                 # first: fx = rho*u
            F[i*3+1] = rhoR[i]*vR[i]*vR[i]+pR[i]     # second: fx = rho*u*u+p
            F[i*3+2] = (ER[i]+pR[i])*vR[i]           # third: fx = (E+p)*u
            

        # for the other cases
        else:
            # Initial solution: (intersection of two linearized integral curves,
            #                    which is actually the upper bound of the solution.)
            pm1 = ( (0.5*(vL[i]-vR[i])*(constants.gamma-1)+aL[i]+aR[i])/
                    (aL[i]*pL[i]**((1-constants.gamma)/constants.gamma*0.5) +
                     aR[i]*pR[i]**((1-constants.gamma)/constants.gamma*0.5)) )**(2*constants.gamma/(constants.gamma-1))

            # Fixed-point iteration to find the pressure and velocity in the middle.
            # (i.e., find the intersection of two nonlinear integral curves.)

            k = 0
            kmax = 100
            
            for k in range(kmax+2):

                mL = massflux(rhoL[i],aL[i],pL[i],pm1)
                mR = massflux(rhoR[i],aR[i],pR[i],pm1)
                pm2 = (mL*pR[i]+mR*pL[i]-mL*mR*(vR[i]-vL[i]))/(mL+mR)

                # Test for fixed point convergence
                if (abs(pm2-pm1) < tol):
                    break

                # Test for max iterations
                k = k + 1
                if (k > kmax):
                    print("Godunov fixed-point iteration did not converge. Exiting.")
                    sys.exit(1)

                # Set old value to new value
                pm1 = pm2

            # Calculate the new fluxes
            mL = massflux(rhoL[i],aL[i],pL[i],pm2)
            mR = massflux(rhoR[i],aR[i],pR[i],pm2)
            vm = (mL*vL[i]+mR*vR[i]-(pR[i]-pL[i]))/(mL+mR)

            # Density in the middle
            r = [rhoL[i],rhoR[i]]
            P = [pL[i],pR[i]]
            gam = (constants.gamma+1)/(constants.gamma-1)
            rm = [None]*2
            for k in range(2):
                if (pm2/P[k] >= 1):
                    rm[k] = r[k]*(1+gam*pm2/P[k]) / (gam+pm2/P[k])
                else:
                    rm[k] = r[k]*( pm2/P[k] )**(1.0/constants.gamma)
                    
            # Contact wave to the right or left?
            if vm >= 0:
                rmI = rm[0]
            else:
                rmI = rm[1]
                
            # Wave speeds at the interface, x/t = 0
            amL = np.sqrt(constants.gamma*pm2/rm[0])
            amR = np.sqrt(constants.gamma*pm2/rm[1])
            SmL = vm - amL
            SmR = vm + amR

            # Sonic case
            if (SmL <= 0) and (SmR >= 0):
                Um2 = rmI*vm
                Um3 = pm2/(constants.gamma-1)+0.5*rmI*vm*vm
            elif (SmL > 0) and ( vL[i] - aL[i] < 0):
                rmI,Um2,Um3 = sonic(vL[i],aL[i],pL[i],vm,amL,vL[i]-aL[i],SmL)
            elif (SmR < 0) and ( vR[i] + aR[i] > 0): 
                rmI,Um2,Um3 = sonic(vR[i],aR[i],pR[i],vm,amR,vR[i]+aR[i],SmR)

            # Compute the flux: evaluate the physical flux at the interface (middle)
            pm   = (constants.gamma-1)*(Um3 - 0.5*Um2*Um2/rmI)
            F[i*3+0] = Um2                  # first: fx = rho*u
            F[i*3+1] = Um2*Um2/rmI + pm     # second: fx = rho*u*u+p
            F[i*3+2] = (Um3 + pm) * Um2/rmI # third: fx = (E+p)*u

    return F


#================================================================================
def massflux(r,c,pQ,pm):
    """ Returns the mass flux used for Godunov's Flux Function

    Katate Masatsuka, February 2009. http://www.cfdbooks.com
    """

    # Smallness
    eps = 1.0e-15

    gam1 = 0.5*(constants.gamma+1)/constants.gamma
    gam2 = 0.5*(constants.gamma-1)/constants.gamma

    if pm/pQ >= 1-eps: # eps to avoid zero-division
        massflux = r*c*np.sqrt(1+gam1*(pm/pQ - 1))
    else:
        massflux = r*c*gam2*(1-pm/pQ) / (1- (pm/pQ)**gam2)

    return massflux


#================================================================================
def sonic(u1,c1,P1,u2,c2,a1,a2):
    """ Returns solutions at sonic points --- used in Godunov's flux
    
    Katate Masatsuka, February 2009. http://www.cfdbooks.com
    """
 
    R1 =  a2/(a2-a1)
    R2 = -a1/(a2-a1)
    us = R1*u1+R2*u2
    cs = R1*c1+R2*c2
    Ps = (cs/c1)**(2.0*constants.gamma/(constants.gamma-1))*P1
    rs = constants.gamma*Ps/(cs*cs)
    
    US1 = rs
    US2 = rs*us
    US3 = Ps/(constants.gamma-1) + 0.5*rs*us*us

    return US1, US2, US3


#================================================================================
def riemann_roe(ul,ur):
    """Returns the Roe interface flux for the Euler equations

    P. L. Roe, Approximate Riemann Solvers, Parameter Vectors and Difference Schemes, Journal of Computational Physics, 43, pp. 357-372.

    Heavily inspired/taken from "I Do Like CFD" website: http://ossanworld.com/cfdbooks/cfdcodes/oned_euler_fluxes_v5.f90

    """

    # Initialize
    F = np.zeros(ul.shape)

    # Primitive variables and sound speeds
    rhoL = ul[0::3]
    vL   = ul[1::3]/rhoL
    EL   = ul[2::3]
    pL   = (constants.gamma-1)*(EL-0.5*rhoL*vL*vL)    
    aL   = np.sqrt(constants.gamma*pL/rhoL)
    HL   = (EL + pL)/rhoL
    
    rhoR = ur[0::3]
    vR   = ur[1::3]/rhoR
    ER   = ur[2::3]
    pR   = (constants.gamma-1)*(ER - 0.5*rhoR*vR*vR)
    aR   = np.sqrt(constants.gamma*pR/rhoR)
    HR   = (ER + pR)/rhoR
    
    # Compute Roe averages
    RT  = np.sqrt(rhoR/rhoL)
    rho = RT*rhoL
    v   = (vL+RT*vR)/(1+RT)
    H   = (HL+RT*HR)/(1+RT)
    a   = np.sqrt((constants.gamma-1)*(H-0.5*v*v))
    dp  = pR - pL

    # Absolute value of Roe eigenvalues
    ws0 = np.fabs(v-a)
    ws1 = np.fabs(v)
    ws2 = np.fabs(v+a)

    # Entropy fix
    Da = np.maximum(0, 4*((vR-aR) - (vL-aL)))
    idx = ws0 < 0.5*Da
    ws0[idx] = ws0[idx]*ws0[idx]/Da[idx] + 0.25*Da[idx]
    Da = np.maximum(0, 4*((vR+aR) - (vL+aL)))
    idx = ws2 < 0.5*Da
    ws2[idx] = ws2[idx]*ws2[idx]/Da[idx] + 0.25*Da[idx]
   
    # Absolute value of Roe eigenvalues * Roe waves strengths
    ws0_dV0 = ws0 * (dp - rho*a*(vR-vL))/(2*a*a)
    ws1_dV1 = ws1 * ((rhoR-rhoL) - dp/(a*a))
    ws2_dV2 = ws2 * (dp + rho*a*(vR-vL))/(2*a*a)

    # Roe Right eigenvectors
    R00 = 1
    R01 = v-a
    R02 = H-v*a
    
    R10 = 1
    R11 = v
    R12 = 0.5*v*v

    R20 = 1
    R21 = v+a
    R22 = H+v*a

    # first: fx = rho*u
    F[0::3] = 0.5*(rhoL*vL + rhoR*vR) \
              -0.5*(ws0_dV0*R00+ 
                    ws1_dV1*R10+ 
                    ws2_dV2*R20)

    # second: fx = rho*u*u+p
    F[1::3] = 0.5*(rhoL*vL*vL+pL  + rhoR*vR*vR+pR) \
              -0.5*(ws0_dV0*R01+ 
                    ws1_dV1*R11+ 
                    ws2_dV2*R21)

    # third: fx = (E+p)*u
    F[2::3] =  0.5*((EL+pL)*vL + (ER+pR)*vR) \
               -0.5*(ws0_dV0*R02+  
                     ws1_dV1*R12+ 
                     ws2_dV2*R22)

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
    p   = (constants.gamma-1)*(E-0.5*rho*v*v)    
    
    # Flux in x-direction
    F[:,0::3] = ug[:,1::3]
    F[:,1::3] = rho*v*v + p
    F[:,2::3] = (E+p)*v
    
    return F

#================================================================================
def sensing(sensors,thresholds,solution):
    """Two sensors to detect contact discontinuities and shocks for the
    Euler equations. See M. T. Henry de Frahan et al. JCP (2015).

    """

    # left/right solution
    ul = solution.u[0,:-solution.N_F]
    ur = solution.u[0,solution.N_F:]
    
    # physical variables on the left and right
    rhoL = ul[0::3]
    vL   = ul[1::3]/rhoL
    EL   = ul[2::3]
    pL   = (constants.gamma-1)*(EL-0.5*rhoL*vL*vL)    
    aL   = np.sqrt(constants.gamma*pL/rhoL)
    HL   = (EL + pL)/rhoL
    
    rhoR = ur[0::3]
    vR   = ur[1::3]/rhoR
    ER   = ur[2::3]
    pR   = (constants.gamma-1)*(ER - 0.5*rhoR*vR*vR)
    aR   = np.sqrt(constants.gamma*pR/rhoR)
    HR   = (ER + pR)/rhoR
    
    # Roe averages
    RT  = np.sqrt(rhoR/rhoL)
    v   = (vL+RT*vR)/(1+RT)
    H   = (HL+RT*HR)/(1+RT)
    a   = np.sqrt((constants.gamma-1)*(H-0.5*v*v))

    # contact wave strength
    drho = rhoR - rhoL
    dp   = pR - pL
    dV2  = drho - dp/(a*a)

    # Discontinuity sensor
    xsi = np.fabs(dV2)/(rhoL+rhoR)
    XSI = 2*xsi/((1+xsi)*(1+xsi))
    idx = np.array(np.where(XSI > thresholds[0]))
    sensors[idx] = 1
    sensors[idx+1] = 1

    # Shock sensor
    phi = np.fabs(dp) / (pL + pR)
    PHI = 2*phi/((1+phi)*(1+phi))
    idx = np.array(np.where( (PHI > thresholds[1]) & (vL-aL > v-a) & (v-a>vR-aR) ))
    sensors[idx] = 2
    sensors[idx+1] = 2
