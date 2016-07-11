#================================================================================
#
# Imports
#
#================================================================================
import sys
import re
import numpy as np
import basis
import copy
import enhance

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
        self.x   = np.empty(self.N_E+1)
        self.xc  = np.empty(self.N_E)
        self.dx  = 0
        self.u   = np.empty([self.basis.N_s, self.N_E])
        self.scaled_minv = np.empty([self.basis.N_s])

        # Manipulation functions
        self.keywords = {
            'system'  : system,
            'printer' : self.print_advection,
            'loader'  : self.load_advection,
            'riemann' : self.riemann_advection,
            'interior_flux' : self.interior_flux_advection,
            'max_wave_speed': self.max_wave_speed_advection,
            'sinewave': self.sinewave,
            'ictest'  : self.ictest,
            'evaluate_face_solution' : self.collocate_faces,
        }

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
    def printer(self,nout,dt):
        """Outputs the solution to a file"""

        print("Solution written to file at step {0:7d} and time {1:e} (current time step:{2:e}).".format(self.n,self.t,dt));
        
        # Call the printing function for the specific problem type
        self.keywords['printer'](nout)


    #================================================================================
    def print_advection(self,nout):
        """Print the solution for linear advection PDE system.

        The format is chosen so that it is easy to plot using matplotlib
        """

        # output file name
        fname = 'u{0:010d}.dat'.format(nout)

        # Concatenate element centroids with the solution (ignore ghost cells)
        xc_u = np.c_[ self.xc, self.u[:,1:-1].transpose()] 

        # Make a descriptive header
        hline = 'n={0:d}, t={1:.18e}, bc_l={2:s}, bc_r={3:s}\nxc'.format(self.n,self.t,self.bc_l,self.bc_r)
        for i in range(self.basis.N_s):
            hline += ', u{0:d}'.format(i)
        
        # Save the data to a file
        np.savetxt(fname, xc_u, fmt='%.18e', delimiter=',', header=hline)

    #================================================================================
    def loader(self,fname):
        """Load the solution from a file"""

        print("Loading file",fname);
        
        # Call the loading function
        self.keywords['loader'](fname)


    #================================================================================
    def load_advection(self,fname):
        """Load the solution for linear advection PDE system.

        """

        # Load the data from the file
        dat = np.loadtxt(fname, delimiter=',')
        self.xc = dat[:,0]
        self.u  = dat[:,1::].transpose()
        
        # Parse the header for the solution time information
        with open(fname,'r') as f:
            line = f.readline()
            line = re.split('=|,',line)
            self.n = int(line[1])
            self.t = float(line[3])
            self.bc_l = line[5].rstrip()
            self.bc_r = line[7].rstrip()

        # Make the basis
        order = self.u.shape[0]-1
        self.basis = basis.Basis(order)

        # Domain specifications
        self.N_E = np.shape(self.u)[1]
        self.dx  = self.xc[1] - self.xc[0]
        A = self.xc[0]  - 0.5*self.dx
        B = self.xc[-1] + 0.5*self.dx
        self.x,self.dx = np.linspace(A, B, self.N_E+1, retstep=True)


    #================================================================================
    def sinewave(self):
        """Sets up the advection of a simple sine wave at a constant velocity"""

        # Domain specifications
        A = -1
        B =  1

        # Initial condition function
        def f(x):
            return np.sin(2*np.pi*x)

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
    def ictest(self):
        """Sets up a test solution."""

        # Domain specifications
        A = -1
        B =  1

        # Initial condition function
        def f(x):
            return np.sin(x-np.pi/3)**3 + 1

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
        """Populate the initial condition, given a function f"""
        
        for e in range(self.N_E):

            # bounds of the element
            a = self.x[e]
            b = self.x[e+1]

            # solution coefficients
            self.u[:,e] = self.basis.projection(a,b,f)


    #================================================================================
    def add_ghosts(self):
        """Add ghost cells to the solution vector"""
        self.u = np.c_[ np.zeros(self.basis.N_s), self.u, np.zeros(self.basis.N_s)] 


    #================================================================================
    def apply_bc(self):
        """Populates the ghost cells with the correct data depending on the BC"""

        # On the left side of the domain
        if self.bc_l is 'periodic':
            self.u[:,0]  = self.u[:,-2]
        else:
            print("{0:s} is an invalid boundary condition. Exiting.".format(self.bc_l))
        
        # On the right side of the domain
        if self.bc_r is 'periodic':
            self.u[:,-1] = self.u[:,1]
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
    def riemann_advection(self,ul,ur):
        """Returns the interface flux for the advection equation (simple upwinding)"""
        return ul
    
    #================================================================================
    def interior_flux(self,ug):
        """Returns the interio flux given the solution at the Gaussian nodes"""
        return self.keywords['interior_flux'](ug)

    #================================================================================
    def interior_flux_advection(self,ug):
        """Returns the interior flux for the advection equation"""
        return ug

    #================================================================================
    def max_wave_speed(self):
        """Returns the maximum wave speed in the domain (based on the cell averages)"""
        return self.keywords['max_wave_speed']()

    #================================================================================
    def max_wave_speed_advection(self):
        """Returns the maximum wave speed for advection"""
        return 1

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
        return self.enhance.face_value(self.u)
