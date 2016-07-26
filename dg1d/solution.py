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
import advection_fluxes
import euler_fluxes

#================================================================================
#
# Class definitions
#
#================================================================================
class Solution:
    'Generate the solution (initialize, ic, mesh, etc)'

    #================================================================================
    def __init__(self,icline,system,order,enhancement_type=''):
        
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
            

    #================================================================================
    def set_manipulation_functions(self,system):
        """Define a dictionary containing the necessary functions"""        

        # Default
        self.keywords = {
            'system'  : system,
            'fields'  : ['u'],
            'riemann' : advection_fluxes.riemann_upwinding,
            'interior_flux' : advection_fluxes.interior_flux,
            'max_wave_speed': advection_fluxes.max_wave_speed,
            'sinewave': self.sinewave,
            'rhobump' : self.rhobump,
            'simplew' : self.simplew,
            'entrpyw' : self.entrpyw,
            'ictest'  : self.ictest,
            'evaluate_face_solution' : self.collocate_faces,
        }
        
        # Modify some of these if solving Euler PDEs
        if system == 'euler':
            self.keywords['fields'] = ['rho','rhou','E']
            self.keywords['riemann'] = euler_fluxes.riemann_rusanov
            self.keywords['interior_flux'] = euler_fluxes.interior_flux
            self.keywords['max_wave_speed'] = euler_fluxes.max_wave_speed
            self.N_F = 3


    #================================================================================
    def printer(self,nout,dt):
        """Outputs the solution to a file

        The format is chosen so that it is easy to plot using matplotlib
        """

        print("Solution written to file at step {0:7d} and time {1:e} (current time step:{2:e}).".format(self.n,self.t,dt));
        
        # output file names
        fnames = self.format_fnames(nout)

        # loop on all the fields
        for field,fname in enumerate(fnames):
        
            # Concatenate element centroids with the solution (ignore ghost cells)
            start =    self.N_F + field
            end   = -2*self.N_F + 1 + field
            step  =    self.N_F
            xc_u = np.c_[ self.xc, self.u[:,start:end:step].transpose()] 

            # Make a descriptive header
            hline = 'n={0:d}, t={1:.18e}, bc_l={2:s}, bc_r={3:s}\nxc'.format(self.n,self.t,self.bc_l,self.bc_r)
            for i in range(self.basis.N_s):
                hline += ', u{0:d}'.format(i)
        
            # Save the data to a file
            np.savetxt(fname, xc_u, fmt='%.18e', delimiter=',', header=hline)


    #================================================================================
    def loader(self,step):
        """Load the solution from a file"""

        print("Loading solution at step",step);

        # File names
        fnames = self.format_fnames(step)

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
    def format_fnames(self,step):
        """Returns a list of file names for a given step"""
        return [field+'{0:010d}.dat'.format(step) for field in self.keywords['fields']]
            
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
        self.u = np.zeros([self.basis.p+1, self.N_E])
        
        # Populate the solution
        self.populate(f)
        
        # Add the ghost cells
        self.add_ghosts()

        # Scale the inverse mass matrix
        self.scaled_minv = self.basis.minv*2.0/self.dx

    #================================================================================
    def rhobump(self):
        """Sets up the advection of a simple density bump at a constant velocity"""

        # Domain specifications
        A = -1
        B =  1

        # Initial condition function
        def f(x):
            gamma = 1.4
            rho = np.sin(2*np.pi*x) + 2
            u   = 1
            p   = 1
            E   = 0.5*rho*u*u + p/(gamma-1)
            return [rho, rho*u, E]

        # Set the boundary condition
        self.bc_l = 'periodic'
        self.bc_r = 'periodic'

        # Number of elements
        self.N_E = int(self.params[0])
        
        # Discretize the domain, get the element edges and the element
        # centroids
        self.x,self.dx = np.linspace(A, B, self.N_E+1, retstep=True)
        self.xc = (self.x[1:] + self.x[:-1]) * 0.5
        
        # Initialize the initial condition
        self.u = np.zeros([self.basis.p+1, self.N_E*self.N_F])
        
        # Populate the solution
        self.populate(f)
        
        # Add the ghost cells
        self.add_ghosts()

        # Scale the inverse mass matrix
        self.scaled_minv = self.basis.minv*2.0/self.dx

    #================================================================================
    def simplew(self):
        """Initial condition for two rarefaction waves moving away from each other.

        See Marcus Lo's thesis p. 164 or my notes 22/7/16. I had to
        fix a lot of typos to make this work.
        """

        
        # Domain specifications
        A = -3
        B =  3

        # Initial condition function
        def f(x):

            # define some constants
            gamma = 1.4
            rho0 = gamma
            p0 = 1
            u0L = -2./np.sqrt(gamma)
            u0R =  2./np.sqrt(gamma)
            M0L = -1
            M0R =  1           
            a0  = 1/np.sqrt(gamma)

            # Velocities vary in different regions
            if x<-1.5 :
                u   = u0L
                rho = rho0 * (1+(gamma-1)/2* (u/a0))**(2.0/(gamma-1))
                p   = p0   * (1+(gamma-1)/2* (u/a0))**(2.0*gamma/(gamma-1))
            elif ((x>=-1.5) and (x<=-0.5)):
                u   = a0*(M0L- u0L/(2*a0) * np.tanh((x+1)/(0.25-(x+1)*(x+1))))
                rho = rho0 * (1+(gamma-1)/2* (u/a0))**(2.0/(gamma-1))
                p   = p0   * (1+(gamma-1)/2* (u/a0))**(2.0*gamma/(gamma-1))
            elif ((x>-0.5) and (x<0.5)):
                u   = 0
                rho = rho0
                p   = p0
            elif ((x>=0.5) and (x<=1.5)):
                u   = a0*(M0R + u0R/(2*a0) * np.tanh((x-1)/(0.25-(x-1)**2)))
                rho = rho0 * (1+(gamma-1)/2* (-u/a0))**(2.0/(gamma-1))
                p   = p0   * (1+(gamma-1)/2* (-u/a0))**(2.0*gamma/(gamma-1))
            elif (x>1.5):
                u   = u0R
                rho = rho0 * (1+(gamma-1)/2* (-u/a0))**(2.0/(gamma-1))
                p   = p0   * (1+(gamma-1)/2* (-u/a0))**(2.0*gamma/(gamma-1))
               
            # Now for the density/pressure/energy fields
            E   = p/(gamma-1) + 0.5*rho*u*u
            
            return [rho, rho*u, E]

        # Set the boundary condition
        self.bc_l = 'zerograd'
        self.bc_r = 'zerograd'

        # Number of elements
        self.N_E = int(self.params[0])
        
        # Discretize the domain, get the element edges and the element
        # centroids
        self.x,self.dx = np.linspace(A, B, self.N_E+1, retstep=True)
        self.xc = (self.x[1:] + self.x[:-1]) * 0.5
        
        # Initialize the initial condition
        self.u = np.zeros([self.basis.p+1, self.N_E*self.N_F])
        
        # Populate the solution
        self.populate(f)
        
        # Add the ghost cells
        self.add_ghosts()

        # Scale the inverse mass matrix
        self.scaled_minv = self.basis.minv*2.0/self.dx

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
            gamma = 1.4
            rho0  = 1
            u0    = 1
            p0    = 1
            A     = 0.2

            # Fill the fields
            rho = rho0 + A * np.sin(np.pi*x)
            u   = u0
            p   = p0
            E   = p/(gamma-1) + 0.5*rho*u*u
            
            return [rho, rho*u, E]

        # Set the boundary condition
        self.bc_l = 'periodic'
        self.bc_r = 'periodic'

        # Number of elements
        self.N_E = int(self.params[0])
        
        # Discretize the domain, get the element edges and the element
        # centroids
        self.x,self.dx = np.linspace(A, B, self.N_E+1, retstep=True)
        self.xc = (self.x[1:] + self.x[:-1]) * 0.5
        
        # Initialize the initial condition
        self.u = np.zeros([self.basis.p+1, self.N_E*self.N_F])
        
        # Populate the solution
        self.populate(f)
        
        # Add the ghost cells
        self.add_ghosts()

        # Scale the inverse mass matrix
        self.scaled_minv = self.basis.minv*2.0/self.dx

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
        self.u = np.zeros([self.basis.p+1, self.N_E])
        
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
        if np.fabs(a) > 1e-14:
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
