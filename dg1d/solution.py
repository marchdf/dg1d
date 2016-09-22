#================================================================================
#
# Imports
#
#================================================================================
import sys
import re
import numpy as np
import itertools
import copy

import basis
import enhance
import advection_physics
import euler_physics
import constants
import sensor

#================================================================================
#
# Class definitions
#
#================================================================================
class Solution:
    'Generate the solution (initialize, ic, mesh, etc)'

    #================================================================================
    def __init__(self,icline,system,order,enhancement_type='',sensor_thresholds=[]):
        
        print("Generating the solution.")

        # A solution contains a basis
        self.basis = basis.Basis(order)

        # It also contains initial condition information
        # parse the input parameters: name and extra parameters
        self.icname = icline.split()[0]
        self.params = icline.split()[1:]

        # And, of course, the solution information itself
        self.t   = 0
        self.n   = 0
        self.N_E = 0
        self.N_F = 1
        self.x   = np.empty(self.N_E+1)
        self.xc  = np.empty(self.N_E)
        self.dx  = 0
        self.u   = np.empty([self.basis.N_s, self.N_E*self.N_F])
        self.scaled_minv = np.empty([self.basis.N_s])

        # Initialize some global constants
        constants.init()
        
        # Manipulation functions
        self.set_manipulation_functions(system)

        # Boundary condition type (left/right)
        self.bc_l = ''
        self.bc_r = ''
                   
        # Apply the initial condition
        try:
            self.keywords[self.icname]()
        except Exception as e:
            print("Invalid initial condition. This will be an empty solution.\n",e)

        # Enhancement (if necessary)
        if (enhancement_type is not ''):
            self.keywords['evaluate_face_solution'] = self.enhanced_faces
            self.enhance = enhance.Enhance(order,enhancement_type,self.u.shape[1])

        # Sensors
        self.issensing = False
        if sensor_thresholds:
            self.issensing = True
            self.sensors = sensor.Sensor(sensor_thresholds,self.N_E+2)

    #================================================================================
    def set_manipulation_functions(self,system):
        """Define a dictionary containing the necessary functions"""        

        # Default
        self.keywords = {
            'system'  : system,
            'fields'  : ['u'],
            'riemann' : advection_physics.riemann_upwinding,
            'interior_flux' : advection_physics.interior_flux,
            'max_wave_speed': advection_physics.max_wave_speed,
            'sensing': advection_physics.sensing,
            'sinewave': self.sinewave,
            'simplew' : self.simplew,
            'entrpyw' : self.entrpyw,
            'acsticw' : self.acsticw,
            'sodtube' : self.sodtube,
            'shuoshe' : self.shuoshe,
            'ictest'  : self.ictest,
            'evaluate_face_solution' : self.collocate_faces,
        }
        
        # Modify some of these if solving Euler PDEs
        if system == 'euler':
            self.keywords['fields'] = ['rho','rhou','E']
            self.keywords['riemann'] = euler_physics.riemann_roe
            self.keywords['interior_flux'] = euler_physics.interior_flux
            self.keywords['max_wave_speed'] = euler_physics.max_wave_speed
            self.keywords['sensing'] =  euler_physics.sensing
            self.N_F = 3


    #================================================================================
    def printer(self,nout,dt):
        """Outputs the solution to a file

        The format is chosen so that it is easy to plot using matplotlib
        """

        print("Solution written to file at step {0:7d} and time {1:e} (current time step:{2:e}).".format(self.n,self.t,dt));
        
        # output file names
        fnames = self.format_fnames(nout,self.keywords['fields'])

        # Descriptive header
        hline = 'n={0:d}, t={1:.18e}, bc_l={2:s}, bc_r={3:s}\nxc'.format(self.n,self.t,self.bc_l,self.bc_r)
        for i in range(self.basis.N_s):
            hline += ', u{0:d}'.format(i)
        
        # loop on all the fields
        for field,fname in enumerate(fnames):
        
            # Concatenate element centroids with the solution (ignore ghost cells)
            start =    self.N_F + field
            end   = -2*self.N_F + 1 + field
            step  =    self.N_F
            xc_u = np.c_[ self.xc, self.u[:,start:end:step].transpose()] 
       
            # Save the data to a file
            np.savetxt(fname, xc_u, fmt='%.18e', delimiter=',', header=hline)

        # Output the sensors if necessary
        if (self.issensing):
            fname = self.format_fnames(nout,['sensor'])
            
            # Concatenate sensor with element centroids
            xc_sen = np.c_[ self.xc, self.sensors.sensors[1:-1]]

            # Quick diagnostic of how many sensors are on
            print("\tsensors on in {0:6.2f}% of the domain".format(np.count_nonzero(xc_sen[:,1])/xc_sen.shape[0]*100))
            
            # Save the data to a file
            np.savetxt(fname[0], xc_sen, fmt='%.18e, %.0d', delimiter=',', header=hline)


    #================================================================================
    def loader(self,step):
        """Load the solution from a file"""

        print("Loading solution at step",step);

        # File names
        fnames = self.format_fnames(step,self.keywords['fields'])

        #
        # Read data from one file
        # 
        self.xc = np.loadtxt(fnames[0], delimiter=',', usecols=(0,))

        # Parse the header and first line for the solution information
        with open(fnames[0],'r') as f:
            line = f.readline()
            line = re.split('=|,',line)
            self.n = int(line[1])
            self.t = float(line[3])
            self.bc_l = line[5].rstrip()
            self.bc_r = line[7].rstrip()

            # get the number of solution coefficients
            line = f.readline()
            N_s  = len(line.split(',')) - 1

        # Make the basis
        order = N_s-1
        self.basis = basis.Basis(order)

        # Domain specifications
        self.N_E = len(self.xc)
        self.dx  = self.xc[1] - self.xc[0]
        A = self.xc[0]  - 0.5*self.dx
        B = self.xc[-1] + 0.5*self.dx
        self.x, self.dx = np.linspace(A, B, self.N_E+1, retstep=True)

        #
        # Read solution data from all the files
        #
        self.u = np.zeros((N_s,self.N_E*self.N_F))
        self.add_ghosts()
        for field,fname in enumerate(fnames):
            print(fname)
        
            # Load the data from the file
            dat = np.loadtxt(fname, delimiter=',')

            # Store the data
            start =    self.N_F + field
            end   = -2*self.N_F + 1 + field
            step  =    self.N_F
            self.u[:,start:end:step]  = dat[:,1::].transpose()


    #================================================================================
    def format_fnames(self,step,fields):
        """Returns a list of file names for a given step"""
        return [field+'{0:010d}.dat'.format(step) for field in fields]
            
    #================================================================================
    def sinewave(self):
        """Sets up the advection of a simple sine wave at a constant velocity"""

        # Domain specifications
        A = -1
        B =  1

        # Initial condition function
        def f(x):
            return [np.sin(2*np.pi*x)]

        # Set the boundary condition
        self.bc_l = 'periodic'
        self.bc_r = 'periodic'

        # Set up the rest of the IC
        self.setup_common_ic(f,A,B)


    #================================================================================
    def simplew(self):
        """Initial condition for two rarefaction waves moving away from each other.

        See the High Order CFD Workshop 1 and 2 problem C1.5.
        See Marcus Lo's thesis p. 164 (lots of typos) 
        Or read my notes 27/7/16. 
        """
        
        # Domain specifications
        A = -4
        B =  4

        # define some constants
        constants.gamma = 3
        
        # Initial condition function
        def f(x):

            # Velocities vary in different regions
            u0 = 2./constants.gamma
            if x<=-1.5 :
                u   = -u0
            elif ((x>-1.5) and (x<-0.5)):
                u   = -1/constants.gamma * (1 - np.tanh((x+1)/(0.25-(x+1)**2))) 
            elif ((x>=-0.5) and (x<=0.5)):
                u   = 0
            elif ((x>0.5) and (x<1.5)):
                u   =  1/constants.gamma * (1 + np.tanh((x-1)/(0.25-(x-1)**2)))
            elif (x>=1.5):
                u   = u0
               
            # Now for the speed of sound/density/pressure/energy fields
            a   = 1 - (constants.gamma-1)/2 * np.fabs(u)
            rho = constants.gamma * (a**(2/(constants.gamma-1)))
            p   = rho*a*a/constants.gamma
            E   = p/(constants.gamma-1) + 0.5*rho*u*u
            
            return [rho, rho*u, E]

        # Set the boundary condition
        self.bc_l = 'zerograd'
        self.bc_r = 'zerograd'

        # Set up the rest of the IC
        self.setup_common_ic(f,A,B)


    #================================================================================
    def entrpyw(self):
        """Initial condition for an entropy wave.
        
        See "I do like CFD" Vol 1 (2nd edition) by Masatsuka on page 240
        """
        
        # Domain specifications
        A = -1
        B =  1

        # Initial condition function
        def f(x):

            # define some constants
            rho0  = 1
            u0    = 1
            p0    = 1
            A     = 0.2

            # Fill the fields
            rho = rho0 + A * np.sin(np.pi*x)
            u   = u0
            p   = p0
            E   = p/(constants.gamma-1) + 0.5*rho*u*u
            
            return [rho, rho*u, E]

        # Set the boundary condition
        self.bc_l = 'periodic'
        self.bc_r = 'periodic'

        # Set up the rest of the IC
        self.setup_common_ic(f,A,B)


    #================================================================================
    def acsticw(self):
        """Initial condition for the propagation of an acoustic wave.
        
        Adapted from Mauro's JCP, Eric's thesis, and Eric's project
        for his CFD class.

        """
        
        # Domain specifications
        A = -1
        B =  1
       
        # Initial condition function
        def f(x):

            # define some constants
            rho0  = 1
            u0    = 0
            p0    = 1/constants.gamma

            epsilon = 1e-4
            h = lambda x: epsilon * (np.cos(0.5*np.pi*x)**8)
            
            # Fill the fields
            rho = rho0 + h(x)
            u   = u0
            p   = p0   + h(x)
            E   = p/(constants.gamma-1) + 0.5*rho*u*u
            
            return [rho, rho*u, E]

        # Set the boundary condition
        self.bc_l = 'periodic'
        self.bc_r = 'periodic'

        # Set up the rest of the IC
        self.setup_common_ic(f,A,B)


    #================================================================================
    def sodtube(self):
        """Initial condition for the Sod shock tube problem.
        
        """
        
        # Domain specifications
        A = -1
        B =  1

        # Initial condition function
        def f(x):

            # define some constants
            constants.gamma = 1.4

            # left state
            if x < 0:
                rho = 1
                u   = 0
                p   = 1.0
                return [rho, rho*u, 1.0/(constants.gamma-1.0)*p + 0.5*rho*u*u]

            # Right state
            elif x >= 0:
                rho = 0.125
                u   = 0
                p   = 0.1
                return [rho, rho*u, 1.0/(constants.gamma-1.0)*p + 0.5*rho*u*u]

        # Set the boundary condition
        self.bc_l = 'zerograd'
        self.bc_r = 'zerograd'

        # Set up the rest of the IC
        self.setup_common_ic(f,A,B)


    #================================================================================
    def shuoshe(self):
        """Initial condition for the Shu-Osher problem
        
        """
        
        # Domain specifications
        A = 0
        B = 10

        # Initial condition function
        def f(x):

            # define some constants
            constants.gamma = 1.4

            # left state
            if x <= 1:
                rho = 3.857143
                u   = 2.629369
                p   = 10.33333
                return [rho, rho*u, 1.0/(constants.gamma-1.0)*p + 0.5*rho*u*u]

            # Right state
            elif x > 1:
                rho = 1.0+0.2*np.sin(5.0*(x-5.0))
                u   = 0
                p   = 1
                return [rho, rho*u, 1.0/(constants.gamma-1.0)*p + 0.5*rho*u*u]

        # Set the boundary condition
        self.bc_l = 'zerograd'
        self.bc_r = 'zerograd'

        # Set up the rest of the IC
        self.setup_common_ic(f,A,B)


    #================================================================================
    def ictest(self):
        """Sets up a test solution."""

        # Domain specifications
        A = -1
        B =  1

        # Initial condition function
        def f(x):
            return [np.sin(x-np.pi/3)**3 + 1]

        # Set the boundary condition
        self.bc_l = 'periodic'
        self.bc_r = 'periodic'

        # Set up the rest of the IC
        self.setup_common_ic(f,A,B)

    #================================================================================
    def setup_common_ic(self,f,A,B):
        """Common stuff always done when setting up an initial condition"""

        # Number of elements
        self.N_E = int(self.params[0])
        
        # Discretize the domain, get the element edges and the element
        # centroids
        self.x,self.dx = np.linspace(A, B, self.N_E+1, retstep=True)
        self.xc = (self.x[1:] + self.x[:-1]) * 0.5
        # self.xg = np.zeros((self.basis.N_G,self.N_E))
        # for e in range(self.N_E):
        #     self.xg[:,e] = self.basis.shifted_xgauss(self.x[e],self.x[e+1])
        
        # Initialize the initial condition
        self.u = np.zeros([self.basis.p+1, self.N_E*self.N_F])
        
        # Populate the solution
        self.populate(f)
        
        # Add the ghost cells
        self.add_ghosts()

        # Scale the inverse mass matrix
        self.scaled_minv = self.basis.minv*2.0/self.dx


    #================================================================================
    def populate(self,f):
        """Populate the initial condition, given a function f

        f is a list of functions, one value for each field
        """
        
        for e in range(self.N_E):

            # bounds of the element
            a = self.x[e]
            b = self.x[e+1]

            for field in range(self.N_F):
                # solution coefficients
                self.u[:,e*self.N_F+field] = self.basis.projection(a,b,f,field)

    #================================================================================
    def add_ghosts(self):
        """Add ghost cells to the solution vector"""
        self.u = np.c_[ np.zeros((self.basis.N_s,self.N_F)), self.u, np.zeros((self.basis.N_s,self.N_F))] 


    #================================================================================
    def apply_bc(self):
        """Populates the ghost cells with the correct data depending on the BC"""

        # On the left side of the domain
        if self.bc_l is 'periodic':
            self.u[:,0:self.N_F]  = self.u[:,-2*self.N_F:-self.N_F]
        elif self.bc_l is 'zerograd':
            self.u[:,0:self.N_F]  = self.u[:,self.N_F:2*self.N_F]
        else:
            print("{0:s} is an invalid boundary condition. Exiting.".format(self.bc_l))
        
        # On the right side of the domain
        if self.bc_r is 'periodic':
            self.u[:,-self.N_F:] = self.u[:,self.N_F:2*self.N_F]
        elif self.bc_r is 'zerograd':
            self.u[:,-self.N_F:] = self.u[:,-2*self.N_F:-self.N_F]
        else:
            print("{0:s} is an invalid boundary condition. Exiting.".format(self.bc_r))
    
    #================================================================================
    def copy(self):
        """Returns a deep copy of a solution"""
        return copy.deepcopy(self)

    #================================================================================
    def copy_data_only(self,other):
        """Copy data u from other solution into the self"""
        self.u = np.copy(other.u)

    #================================================================================
    def smart_axpy(self,a,x):
        """Adds a*x to u only if a is non-zero"""
        if np.fabs(a) > 1e-15:
            self.u += a*x

    #================================================================================
    def riemann(self,ul,ur):
        """Returns the flux at an interface by calling the right Riemann solver"""
        return self.keywords['riemann'](ul,ur)
    
    #================================================================================
    def interior_flux(self,ug):
        """Returns the interio flux given the solution at the Gaussian nodes"""
        return self.keywords['interior_flux'](ug)

    #================================================================================
    def max_wave_speed(self):
        """Returns the maximum wave speed in the domain (based on the cell averages)"""
        return self.keywords['max_wave_speed'](self.u)

    #================================================================================
    def collocate(self):
        """Collocate the solution to the Gaussian quadrature nodes"""
        return np.dot(self.basis.phi, self.u)

    #================================================================================
    def evaluate_faces(self):
        """Evaluate the solution at the cell edges/faces"""

        # Call the correct face evaluation procedure
        return self.keywords['evaluate_face_solution']()

    #================================================================================
    def collocate_faces(self):
        """Collocate the solution to the cell edges/faces"""
        return np.dot(self.basis.psi, self.u)

    #================================================================================
    def enhanced_faces(self):
        """Get the value of the enhanced solution at the faces"""
        return self.enhance.face_value(self.u,self.N_F)
