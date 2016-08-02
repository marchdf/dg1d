#================================================================================
#
# Imports
#
#================================================================================
import sys
import numpy as np
import rk_coeffs as rkc

#================================================================================
#
# Function definitions
#
#================================================================================

#================================================================================
def integrate(solution,deck,dgsolver,limiter):
    """Integrate in time using an RK scheme"""

    if deck.rk == 'low_storage_rk4':
        low_storage_rk4(solution,deck,dgsolver)

    else:
        if deck.rk   == 'rk3':
            coeffs, alphas, betas = rkc.get_rk3_coefficients()

        elif deck.rk == 'rk4':
            coeffs, alphas, betas = rkc.get_rk4_coefficients()

        elif deck.rk == 'rk5':
            coeffs, alphas, betas = rkc.get_rk5_coefficients()

        elif deck.rk == 'rk6':
            coeffs, alphas, betas = rkc.get_rk6_coefficients()

        elif deck.rk == 'rk8':
            coeffs, alphas, betas = rkc.get_rk8_coefficients()

        elif deck.rk == 'rk10':
            coeffs, alphas, betas = rkc.get_rk10_coefficients()

        elif deck.rk == 'rk12':
            coeffs, alphas, betas = rkc.get_rk12_coefficients()

        elif deck.rk == 'rk14':
            coeffs, alphas, betas = rkc.get_rk14_coefficients()

        else:
            print('Unrecognized RK option, default to RK4')
            coeffs, alphas, betas = rkc.get_rk4_coefficients()

        classic_rk(solution,deck,dgsolver,limiter,coeffs,alphas,betas)


#================================================================================
def classic_rk(solution,deck,dgsolver,limiter,coeffs,alphas,betas):
    """Integrate in time using the classic RK4 scheme"""

    # Initialize storage variables
    K  = [np.zeros(solution.u.shape) for _ in range(len(coeffs))]
    us = solution.copy()
    uk = solution.copy()
   
    # Output time array (ignore the start time)
    nout = 0
    tout_array = iter(np.linspace(solution.t,deck.finaltime,deck.nout)[1:]) 

    # Flags
    done = False

    # Write the initial condition to file
    solution.printer(0,0.0)
    nout += 1
    tout = next(tout_array)

    # main RK loop
    while (not done):

        # Get the next time step
        dt,output,done = get_next_time_step(solution,tout,deck.cfl,deck.finaltime)

        # Store the solution at the previous step: us = u
        us.copy_data_only(solution)
        
        # RK inner loop
        for k,(c,alpha) in enumerate(zip(coeffs,alphas)):

            # Get the solution at this sub-time step:
            # u_k = u_0 + \Delta t \sum_{k=0}^{n-1} \beta_{k,j} f(t_j,u_j)
            # t_k = t_0 + \alpha_k \Delta t
            uk.copy_data_only(us)
            for j,beta in enumerate(betas[k,:k]):
                uk.smart_axpy(beta,K[j])
            uk.t += alpha * dt

            # Limit solution if necessary
            if k > 0: limiter.limit(uk)
            
            # Evaluate and store the solution increment: K_k = \Delta t  f(t_k, u_k)
            K[k] = dt*dgsolver.residual(uk)

            # Weighted sum of the residuals
            solution.smart_axpy(c,K[k])

        # Limit solution if necessary
        limiter.limit(solution)

        # Update the current time
        solution.t += dt
        solution.n += 1

        # Output the solution if necessary
        if output:
            solution.printer(nout,dt)
            if not done:
                nout += 1
                tout = next(tout_array)

        
    
#================================================================================
def low_storage_rk4(solution,deck,dgsolver):
    """Integrate in time using the classic RK4 scheme with low storage algorithm"""

    # Initialize storage variables
    us    = solution.copy()
    ustar = solution.copy()
    du = np.zeros(solution.u.shape)
    
    # Output time array (ignore the start time)
    nout = 0
    tout_array = iter(np.linspace(solution.t,deck.finaltime,deck.nout)[1:]) 

    # Flags
    done = False

    # RK4 coefficients
    betas  = [0.0, 0.5, 0.5, 1.0]
    gammas = [1./6., 1./3., 1./3., 1./6.];

    # Write the initial condition to file
    solution.printer(0,0.0)
    nout += 1
    tout = next(tout_array)
    
    # Main RK4 loop
    while (not done):
            
        # Get the next time step
        dt,output,done = get_next_time_step(solution,tout,deck.cfl,deck.finaltime)
        
        # Store the solution at the previous step: us = u
        us.copy_data_only(solution)

        # RK inner loop
        for beta,gamma in zip(betas,gammas):

            # Calculate the star quantities
            ustar.copy_data_only(us)
            ustar.smart_axpy(beta,du)
            ustar.t += beta*dt
            
            # Limit solution if necessary
            if k > 0: limiter.limit(ustar)
            
            # Calculate the solution increment (=dt*residual)
            du = dt * dgsolver.residual(ustar)
            
            # Update the solution
            solution.smart_axpy(gamma,du)

        # Limit solution if necessary
        limiter.limit(solution)
        
        # Update the current time
        solution.t += dt
        solution.n += 1

        # Output the solution if necessary
        if output:
            solution.printer(nout,dt)
            if not done:
                nout += 1
                tout = next(tout_array)
                

#================================================================================
def get_next_time_step(solution,tout,cfl,tf):
    """Returns the next time step and output/done flags"""

    # Time step from CFL
    dt = cfl_time_step(solution,cfl)

    # Sanity check for this timestep
    sanity_check_dt(dt,solution.n,solution.t)

    # Return the time step and output/done flags
    return adjust_for_output(dt,solution.t,tf,tout)

    
#================================================================================
def cfl_time_step(solution,cfl):
    """Given the solution and the CFL condition, determine the next time step size

    """

    # Get the maximum wave speed in the domain
    v = solution.max_wave_speed()

    # Return the time step
    return solution.dx*cfl/( v * (2*solution.basis.p+1) )
    #return (solution.dx**2)*cfl/( v * (2*solution.basis.p+1) )
    
    
#================================================================================
def sanity_check_dt(dt,n,t):
    """Make sure the next time step is not absurd

    """

    if dt < 1e-14:
        print("Next time step is too small ({0:e}<1e-14). Exiting at step {1:7d} and time {2:e}.\n".format(dt,n,t))
        sys.exit()
        
    if np.isnan(dt):
        print("Time step is NaN. Exiting at step {0:7d} and time {1:e}.\n".format(n,t))
        sys.exit()

#================================================================================
def adjust_for_output(dt,t,tf,tout):
    """Returns a new time step and true output flag if you need to output
    (also checks if you are done with the time integration)
        
    """
    #eps = 1e-14

    # If we reached the final time (or almost)
    if dt > tf-t:
        return tf-t, True, True
        
    # If we reached an output time
    #elif dt > (tout-t) - eps:
    elif dt > tout-t:
        return tout-t, True, False    

    else:
        return dt, False, False
    
