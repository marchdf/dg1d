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
def integrate(solution,deck,dgsolver):
    """Integrate in time using RK4"""

    # Initialize storage variables
    us    = solution.copy()
    ustar = solution.copy()
    du    = solution.copy()
    fstar = np.zeros(solution.u.shape)
    
    # Output time array (ignore the start time)
    tout_array = iter(np.linspace(solution.t,deck.finaltime,deck.nout)[1:]) 

    # Flags
    done = False

    # RK4 coefficients
    beta = [0.0, 0.5, 0.5, 1.0]
    gamma= [1.0/6.0, 2.0/6.0, 2.0/6.0, 1.0/6.0];

    # Write the initial condition to file
    solution.printer(0.0)
    tout = next(tout_array)
    
    # Main RK4 loop
    while (not done):
            
        # Get the next time step
        dt,output,done = get_next_time_step(solution,tout,deck.cfl,deck.finaltime)
        
        # Store the solution at the previous step: us = u
        us = solution.copy()

        # RK inner loop
        for k in range(0,len(beta)):

            # Calculate the star quantities
            ustar = us.copy();
            ustar.axpy(beta[k], du)
            ustar.t += beta[k]*dt
            
            # Limit solution if necessary
            
            # Calculate the residual
            fstar = dgsolver.residual(ustar)
            
            # Calculate the solution increment
            du.u = dt*fstar

            # Update the solution
            solution.axpy(gamma[k],du)

        # Update the current time
        solution.t += dt
        solution.n += 1

        # Output the solution if necessary
        if output:
            solution.printer(dt)
            if not done:
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

    # TODO
    
    return 0.1
    
    
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
    eps = 1e-14

    # If we reached the final time (or almost)
    if dt > (tf-t) - eps:
        return tf-t, True, True
        
    # If we reached an output time
    elif dt > (tout-t) - eps:
        return tout-t, True, False    

    else:
        return dt, False, False
    
