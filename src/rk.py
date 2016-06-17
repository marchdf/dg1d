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
def integrate(u,deck):
    """Integrate in time using RK4
    
    """

    # initialize storage variables
    us    = np.zeros(u.shape)
    ustar = np.zeros(u.shape)
    fstar = np.zeros(u.shape)
    du    = np.zeros(u.shape)
    
    # Counters
    count = 0 # count output steps
    n     = 0 # count time steps
    
    # time variables
    t    = 0
    tf   = deck.finaltime
    nout = deck.nout
    tout_array = np.linspace(t,tf,nout)
    dt   = 0
    cfl  = deck.cfl

    # Flags
    done = False

    # RK4 coefficients
    beta = [0.0, 0.5, 0.5, 1.0]
    gamma= [1.0/6.0, 2.0/6.0, 2.0/6.0, 1.0/6.0];

    # Write the initial condition to file
    print("Initial condition written to output file.")
    # TODO printer.print(u)
    count += 1
    tout = tout_array[count]

    
    # Main RK4 loop
    while (not done):
            
        # Get the next time step
        dt,output,done =get_next_time_step(u,cfl,dt,n,t,tf,tout)
        
        # Store the solution at the previous step: us = u
        np.copyto(us,u)

        # RK inner loop
        for k in range(0,len(beta)):

            # Calculate the star quantities
            ustar = us + beta[k]*du
            tstar = t  + beta[k]*dt
            
            # Limit solution if necessary
            
            # Calculate the residual
            # TODO fstar = -u
            
            # Calculate the solution increment
            du = dt*fstar

            # Update the solution
            u = u + gamma[k]*du

        # Update the current time
        t = t+dt
        n += 1

        # Output the solution if necessary
        if output:
            print("Solution written to file at step {0:7d} and time {1:e} (current time step:{2:e}).\n".format(n,t,dt));
            # TODO printer.print(u)
            if not done:
                count += 1
                tout = tout_array[count]

            

#================================================================================
def get_next_time_step(u,cfl,dt,n,t,tf,tout):
    """Returns the next time step and output/done flags"""

    # Time step from CFL
    dt = cfl_time_step(u,cfl)

    # Sanity check for this timestep
    sanity_check_dt(dt,n,t)

    # Return the time step and output/done flags
    return adjust_for_output(dt,t,tf,tout)

    
#================================================================================
def cfl_time_step(u,cfl):
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
    
