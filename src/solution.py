#================================================================================
#
# Imports
#
#================================================================================
import sys
import re
import numpy as np
import basis

#================================================================================
#
# Class definitions
#
#================================================================================
class Solution:
    'Generate the solution (initialize, ic, mesh, etc)'

    #================================================================================
    def __init__(self,icline,system,order):
        
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
        self.u   = np.empty([self.basis.p+1, self.N_E])

        # Manipulation functions
        self.printers = {
            'advection' : self.print_advection
        }
        self.loaders = {
            'advection' : self.load_advection
        }
        self.type = system
        

    #================================================================================
    def printer(self,dt):
        """Outputs the solution to a file"""

        print("Solution written to file at step {0:7d} and time {1:e} (current time step:{2:e}).\n".format(self.n,self.t,dt));
        
        # Call the printing function
        self.printers[self.type]()


    #================================================================================
    def print_advection(self):
        """Print the solution for linear advection PDE system.

        The format is chosen so that it is easy to plot using matplotlib
        """

        # output file name
        fname = 'u{0:010d}.dat'.format(self.n)

        # Concatenate element centroids with the solution
        xc_u = np.c_[ self.xc, self.u.transpose()] 

        # Make a descriptive header
        hline = 'n={0:d}, t={0:.18e}\nxc'.format(self.n,self.t)
        for i in range(self.basis.N_s):
            hline += ', u{0:d}'.format(i)
        
        # Save the data to a file
        np.savetxt(fname, xc_u, fmt='%.18e', delimiter=',', header=hline)

    #================================================================================
    def loader(self,fname):
        """Load the solution from a file"""

        print("Loading file",fname);
        
        # Call the loading function
        self.loaders[self.type](fname)


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
            self.n = float(line[1])
            self.t = float(line[3])

        # Domain specifications
        self.N_E = np.shape(self.u)[1]
        self.dx  = self.xc[1] - self.xc[0]
        A = self.xc[0]  - 0.5*self.dx
        B = self.xc[-1] + 0.5*self.dx
        self.x,self.dx = np.linspace(A, B, self.N_E+1, retstep=True)


    #================================================================================
    def apply_ic(self):
        """Apply the initial condition"""

        # Set the time/time step to zero
        self.t = 0
        self.n = 0

        # Setup dictionary of initial conditions according to
        # http://stackoverflow.com/questions/2283210/python-function-pointer
        ics = {
            'sinewave': self.sinewave,
        }

        # Call the initial condition function
        try:
            ics[self.icname]()
        except Exception as e:
            print("Invalid initial condition. Exiting.\n",e)
            sys.exit()

    #================================================================================
    def sinewave(self):
        """Sets up the advection of a simple sine wave at a constant velocity"""

        # Number of elements
        self.N_E = int(self.params[0])

        # Domain specifications
        A = -2
        B =  2
        L = B-A

        # Discretize the domain, get the element edges and the element
        # centroids
        self.x,self.dx = np.linspace(A, B, self.N_E+1, retstep=True)
        self.xc = (self.x[1:] + self.x[:-1]) * 0.5
        # self.xg = np.zeros((self.basis.N_G,self.N_E))
        # for e in range(self.N_E):
        #     self.xg[:,e] = self.basis.shifted_xgauss(self.x[e],self.x[e+1])
        
        # Initialize the initial condition
        self.u = np.zeros([self.basis.p+1, self.N_E])

        # Initial condition function
        def f(x):
            return np.sin(2*np.pi*x)

        # Populate the solution
        self.populate(f)


    #================================================================================
    def populate(self,f):
        """Populate the initial condition, given a function f"""
        
        for e in range(self.N_E):

            # bounds of the element
            a = self.x[e]
            b = self.x[e+1]

            # solution coefficients
            self.u[:,e] = self.basis.projection(a,b,f)
            
