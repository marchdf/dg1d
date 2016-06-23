#================================================================================
#
# Imports
#
#================================================================================
import numpy as np
import sys

import matplotlib.pyplot as plt

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

        # Initialize variables
        self.ug = np.zeros((solution.u.shape))
        self.uf = np.zeros((2,solution.u.shape[1]))


    #================================================================================
    def residual(self,solution):
        """Calculates the residual for the DG method
        
        residual = Minv*(Q+F)
        where Minv is the inverse mass matrix
              Q is the edge fluxes
              F is the interior flux
        """

        # Apply boundary conditions
        solution.apply_bc()
      
        # Collocate the solution to the Gaussian nodes
        self.ug = collocate(solution)

        # Collocate to the cell edge values
        self.uf = collocate_faces(solution)
                
        # for e in range(solution.N_E):
        #     a = solution.x[e]
        #     b = solution.x[e+1]
        #     xg = 0.5*(b-a)*solution.basis.x + 0.5*(b+a)
        #     plt.plot(xg,self.ug[:,e+1],'bo')
        #     plt.plot([a,b],self.uf[:,e+1],'ro')
        
            
        # xe = np.linspace(-2,2,200)
        # fe = np.sin(2*np.pi*xe)
        # plt.plot(xe,fe,'k')

        # plt.show()
        # sys.exit()

        
        # Evaluate the interior fluxes
        F = 1.0*self.ug
        
        # Integrate the interior fluxes
        F = integrate_interior_flux(solution.basis.dphi_w,F)
        
        # Evaluate the edge fluxes
        q = solution.riemann(self.uf[1,:-1], # left
                             self.uf[0,1:])  # right

        # Add the interior and edge fluxes
        add_interior_face_fluxes(F,q)
        
        # Multiply by the inverse mass matrix
        F = inverse_mass_matrix_multiply(F, solution.scaled_minv)
       
        return F

#================================================================================
def collocate(solution):
    """Collocate the solution to the Gaussian quadrature nodes"""
    return np.dot(solution.basis.phi, solution.u)

#================================================================================
def collocate_faces(solution):
    """Collocate the solution to the cell edges/faces"""
    return np.dot(solution.basis.psi, solution.u)

#================================================================================
def integrate_interior_flux(D,F):
    """Integrate the interior fluxes F, given the basis gradients, D"""
    return np.dot(D,F)

#================================================================================
def add_interior_face_fluxes(F,qp):
    """Adds the face flux contributions to the interior fluxes.
    
    F is the interior flux matrix
    qp is the flux on the right edge of a cell (j+1/2)
    """

    # The edge flux matrix alternates the difference and sum (because
    # of that (A-1)^m factor)
    Q       = np.array([np.diff(qp)]*F.shape[0])
    Q[1::2] = qp[:-1] + qp[1:]
    
    # Only add to the inside of the flux matrices (ignore the ghost
    # fluxes)
    F[:,1:-1] -= Q

#================================================================================
def inverse_mass_matrix_multiply(F,minv):
    """Returns the multiplication of the total fluxes by the inverse mass matrix

    Implementation idea from http://stackoverflow.com/questions/18522216/multiplying-across-in-a-numpy-array        
    """
    return np.einsum('ij,i->ij',F,minv)


