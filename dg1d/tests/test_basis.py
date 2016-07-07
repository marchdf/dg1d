#================================================================================
#
# Imports
#
#================================================================================
import unittest
import basis
import numpy as np
import numpy.testing as npt

from numpy.polynomial import Polynomial as P   # import the Legendre class
from numpy.polynomial import Legendre as L     # import the Legendre class

#================================================================================
#
# Class definitions
#
#================================================================================
class BasisTestCase(unittest.TestCase):
        """Tests for `basis.py`."""

        #================================================================================
        # eval_basis_gauss
        def test_eval_basis_gauss(self):
            """Are the basis correctly evaluated at the Gaussian quadrature nodes?"""
            order = 3
            test_basis = basis.Basis(order)
            phi,dphi_w = test_basis.eval_basis_gauss()
            npt.assert_array_almost_equal(phi, np.array([[ 1. , -0.861136312,  0.612333621, -0.304746985],
                                                         [ 1. , -0.339981044, -0.326619335,  0.411728   ],
                                                         [ 1. ,  0.339981044, -0.326619335, -0.411728   ],
                                                         [ 1. ,  0.861136312,  0.612333621,  0.304746985]]), decimal = 9)
            
            npt.assert_array_almost_equal(dphi_w, np.array([[ 0.         ,  0.         ,  0.         ,  0.         ],
                                                            [ 0.347854845,  0.652145155,  0.652145155,  0.347854845],
                                                            [-0.898651315, -0.665150971,  0.665150971,  0.898651315],
                                                            [ 1.412870929, -0.412870929, -0.412870929,  1.412870929]]), decimal = 9)

        #================================================================================
        # eval_basis_edges
        def test_eval_basis_edges(self):
            """Are the basis correctly evaluated at the edges?"""
            order = 3
            test_basis = basis.Basis(order)
            psi = test_basis.eval_basis_edges()
            npt.assert_array_almost_equal(psi, np.array([[ 1.0, -1.0, 1.0, -1.0],
                                                         [ 1.0,  1.0, 1.0,  1.0]]), decimal = 9)


        #================================================================================
        # mass_matrix
        def test_mass_matrix(self):
            """Is the mass matrix and its inverse correct?"""
            order = 3
            test_basis = basis.Basis(order)
            m,minv = test_basis.mass_matrix()
            npt.assert_array_almost_equal(m, [2,2.0/3.0,2.0/5.0,2.0/7.0], decimal = 9)
            npt.assert_array_almost_equal(minv, [1./2.,3./2.,5./2.,7./2.], decimal = 9)


        #================================================================================
        # projection
        def test_projection0(self):
            """Is the projection correct for f(x) = x^2 on [-1,1]"""
            order = 3
            test_basis = basis.Basis(order)
            f = lambda x: x*x
            c = test_basis.projection(-1,1,f)
            npt.assert_array_almost_equal(c,[1./3,0,2./3,0])

        def test_projection1(self):
            """Is the projection correct for f(x) = sin(x) on [0,0.2]"""
            f = lambda x: np.sin(x)
            a = 0
            b = 0.2
            order = 3
            test_basis = basis.Basis(order)
            c = test_basis.projection(a,b,f)
            npt.assert_array_almost_equal(c,np.array([9.966711e-02, 9.940095e-02, -3.325404e-04, -6.630519e-05]))



        #================================================================================
        # shift_legendre_polynomial
        def test_shift_legendre_polynomial(self):
                """Is the shifting of Legendre polynomials correct"""
                
                # Given a Legendre Polynomial: L(x) = 0.5*(3x^2 -1)
                l = L.basis(2)
                
                # Evaluate L(x+2)
                ls = basis.shift_legendre_polynomial(l,2)
                
                # This should be equal to 5.5 + 6x + 1.5 x^2
                npt.assert_array_almost_equal(ls.convert(kind=P).coef,np.array([5.5,6,1.5]),decimal=9)



        #================================================================================
        # integrate_legendre_product
        def test_integrate_legendre_product(self):
                """Is the integral of the product of two Legendre polynomials correct"""
                
                # Given a Legendre Polynomial: L(x) = 0.5*(3x^2 -1)
                l1 = L.basis(2)
                
                # Evaluate L(x+2)
                l2 = basis.shift_legendre_polynomial(l1,2)
                
                # The integral of l1*l2 over [-1,1] = 0.4
                self.assertAlmostEqual(basis.integrate_legendre_product(l1,l2), 0.4)

            
if __name__ == '__main__':
    unittest.main()
                
