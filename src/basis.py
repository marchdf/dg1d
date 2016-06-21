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

        print("Generating the basis functions.")

        # polynomial order
        self.p   = order

        # number of coefficients in solution
        self.N_s = order+1

        # Get the Gaussian nodes and weights. We want to integrate
        # exactly at most \int phi^(p) phi^(p) dx which is a
        # polynomial of degree 2p. So we want a Gaussian quadrature
        # rule of p+1 (to integrate exactly a polynomial of 2p+2 and
        # less).       
        self.x, self.w = leg.leggauss(self.p+1)
        self.N_G = len(self.x)
        
        # Construct useful basis matrices
        self.phi, self.dphi_w = self.eval_basis_gauss()

        # Construct the matrix to evaluate a solution at the cell edges
        self.psi = self.eval_basis_edges()
        
        # Construct the (unscaled) mass matrix and its inverse
        self.m, self.minv = self.mass_matrix()


    #================================================================================
    def eval_basis_gauss(self):
        """Evaluate the basis at the Gaussian quadrature nodes.

        phi will be used to transform Legendre solution coefficients
        to the solution evaluated at the Gaussian quadrature nodes.
        
        dphi_w will be used for the interior flux integral

        """
        phi    = np.zeros((len(self.x),self.N_s))
        dphi_w = np.zeros((len(self.x),self.N_s))

        for n in range(self.N_s):

            # Get the Legendre polynomial of order n and its gradient
            l  = L.basis(n)
            dl = l.deriv()

            # Evaluate the basis at the Gaussian nodes
            phi[:,n] = leg.legval(self.x,l.coef)

            # Evaluate the gradient at the Gaussian nodes and multiply by the weights
            dphi_w[:,n] = leg.legval(self.x,dl.coef)*self.w

        return phi, dphi_w


    #================================================================================
    def eval_basis_edges(self):
        """Evaluate the basis at the Gaussian quadrature nodes.

        psi will be used to evaluate a solution at the cell edges.
        
        """
        n    = np.arange(0,self.N_s)
        psi  = [(-1)**n, np.ones(self.N_s)]
        return psi
        
    
    #================================================================================
    def mass_matrix(self):
        """Return the mass matrix and inverse mass matrix (of the Legendre
           polynomial basis)
    
        \int_{-1}^1 phi_n phi_n dx, where phi_n is the Legendre polynomial

        """
        idx = np.arange(self.N_s)
        return 2.0/(2*idx+1), (2*idx+1)/2.0


    #================================================================================
    def projection(self,a,b,f):
        """Given a function, returns the approximating polynomial in the interval [a,b]
    
        We are basically projecting the function into the Legendre basis space
        
        f(x) \approx p(x) = \sum_{n=0}^p \frac{<f,phi^(n)>}{<phi^(n),phi^(n)>} phi^(n)
        where <f,g> = int_a^b f(x) g(x) dx
              phi^(n) is the local Legendre polynomial
        
        We also know that 
             int_a^b f(x) g(x) dx \approx \sum_{k=0}^{p-1} w_k dx/2 f(\frac{b-a}{2} x_k + \frac{b+a}{2}) g(\frac{b-a}{2} x_k + \frac{b+a}{2})
             phi^(n)(\frac{b-a}{2} x_k + \frac{b+a}{2}) = L^n(x_k) (where L is the Legendre polynomial on [-1,1])

        """

        # Evaluate the function at the local Gaussian nodes
        xg = self.shifted_xgauss(a,b)
        fgauss = f(xg)

        # There are two ways to do the projection.
        # 1) Manually
        # dx = b-a
        # coef = np.zeros(self.N_s)
        # for n in np.arange(self.N_s):
        #     num = 0.5*dx*sum( self.w * fgauss * self.phi[:,n])
        #     den = 0.5*dx*sum( self.w * self.phi[:,n] * self.phi[:,n])
        #     coef[n] = num/den

        # 2) Using the built-in fit function to fit the data at the
        # Gaussian quadrature nodes. 
        coef = leg.legfit(self.x,fgauss,self.p)

        return coef

    #================================================================================
    def shifted_xgauss(self,a,b):
        """Return the Gaussian nodes in the interval [a,b]"""
        return 0.5*(b-a)*self.x + 0.5*(b+a)
