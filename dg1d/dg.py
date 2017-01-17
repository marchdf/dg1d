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
        
        # Initialize variables
        self.ug = np.zeros((solution.u.shape))
        self.uf = np.zeros((2,solution.u.shape[1]))
        self.q  = np.zeros((solution.N_E+1)*solution.N_F)
        self.F  = np.zeros(solution.u.shape)
        self.Q  = np.zeros((self.F.shape[0],solution.N_E*solution.N_F))

    #================================================================================
    def residual(self,solution):
        """Calculates the residual for the DG method
        
        residual = Minv*(Q+F)

        where Minv is the inverse mass matrix, Q is the edge fluxes, F
        is the interior flux.

        """

        # Apply boundary conditions
        solution.apply_bc()
      
        # Collocate the solution to the Gaussian nodes
        self.ug = solution.collocate()

        # Evaluate the solution at the cell face
        self.uf = solution.evaluate_faces()
       
        # Evaluate the interior fluxes
        self.F = solution.interior_flux(self.ug)
        
        # Integrate the interior fluxes
        self.integrate_interior_flux(solution.basis.dphi_w)
        
        # Evaluate the edge fluxes
        self.q = solution.riemann(self.uf[1,:-solution.N_F], # left
                                  self.uf[0,solution.N_F:])  # right

        # Add the interior and edge fluxes
        self.add_interior_face_fluxes(solution.N_F)
        
        # Multiply by the inverse mass matrix
        self.inverse_mass_matrix_multiply(solution.scaled_minv)
       
        return self.F


    #================================================================================
    def integrate_interior_flux(self,D):
        """Integrates the interior fluxes F, given the basis gradients, D"""
        self.F = np.dot(D,self.F)

    #================================================================================
    def add_interior_face_fluxes(self, N_F):
        """Adds the face flux contributions to the interior fluxes.
        
        """
        
        # The edge flux matrix alternates the difference and sum (because
        # of that (A-1)^m factor)
        self.Q[::2,]  = self.q[N_F:]  - self.q[:-N_F]
        self.Q[1::2,] = self.q[:-N_F] + self.q[N_F:]
        
        # Only add to the inside of the flux matrices (ignore the ghost
        # fluxes)
        self.F[:,N_F:-N_F] -= self.Q

    #================================================================================
    def inverse_mass_matrix_multiply(self,minv):
        """Returns the multiplication of the total fluxes by the inverse mass matrix

        Implementation idea from http://stackoverflow.com/questions/18522216/multiplying-across-in-a-numpy-array        
        """
        self.F = np.einsum('ij,i->ij',self.F,minv)
