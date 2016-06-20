#================================================================================
#
# Imports
#
#================================================================================
import numpy as np
from numpy.polynomial import Polynomial as P
from numpy.polynomial import legendre as leg # import the Legendre functions
from numpy.polynomial import Legendre as L   # import the Legendre class

#================================================================================
#
# Class definitions
#
#================================================================================
class Basis:
    'Generate the basis functions, gradients and Gaussian quadrature'

    #================================================================================
    def __init__(self,order):

        self.order = order

        # Get the Gaussian nodes and weights
        self.x, self.w = leg.leggauss(order)
        
        # Construct useful basis matrices
        #
        self.phi, self.dphi_w = self.eval_basis_gauss(order)
            
        # Construct the (unscaled) mass matrix and its inverse
        self.m, self.minv = self.mass_matrix(order)


    #================================================================================
    def eval_basis_gauss(self,order):
        """Evaluate the basis at the Gaussian quadrature nodes.

        phi will be used to transform Legendre solution coefficients
        to the solution evaluated at the Gaussian quadrature nodes.
        
        dphi_w will be used for the interior flux integral

        """
        phi    = np.zeros((order,order+1))
        dphi_w = np.zeros((order,order+1))

        for n in range(order+1):

            # Get the Legendre polynomial of order n and its gradient
            l  = L.basis(n)
            dl = l.deriv()

            # Evaluate the basis at the Gaussian nodes
            phi[:,n] = leg.legval(self.x,l.coef)

            # Evaluate the gradient at the Gaussian nodes and multiply by the weights
            dphi_w[:,n] = leg.legval(self.x,dl.coef)*self.w

        return phi, dphi_w

    
    #================================================================================
    def mass_matrix(self,order):
        """Given a polynomial order, return the mass matrix and inverse mass
           matrix (of the Legendre polynomial basis)

        """
        idx       = np.arange(order+1)
        return 1.0/(2*idx+1), 2*idx+1
