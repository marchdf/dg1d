#================================================================================
#
# Imports
#
#================================================================================
import numpy as np

#================================================================================
#
# Function definitions
#
#================================================================================
class DG:
    'DG method solver'

    #================================================================================
    def __init__(self,solution):

        print("Initializing the DG solver.")
        
        # Type of Riemann solver to use
        self.riemann = 'upw'

        # Variables
        self.ug = np.zeros((solution.basis.N_s,solution.N_E))
        self.uf = np.zeros((2,solution.N_E))

    #================================================================================
    def residual(self,solution):
        """Calculates the residual for the DG method
        
        residual = Minv*(Q+F)
        where Minv is the inverse mass matrix
              Q is the edge fluxes
              F is the interior flux
        """

        # Collocate the solution to the Gaussian nodes
        self.ug = self.collocate(solution)
        
        # Collocate to the cell edge values
        
        # Evaluate the interior fluxes
        
        # Evaluate the edge fluxes
        
        # Integrate the interior fluxes
        
        # Add the interior and edge fluxes
        
        # Multiply by the inverse mass matrix
        

        return -solution.u

        
    #================================================================================
    def collocate(self,solution):
        """Collocate the solution to the Gaussian quadrature nodes"""
        return np.dot(solution.basis.phi, solution.u)
