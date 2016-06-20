#================================================================================
#
# Imports
#
#================================================================================
import unittest
import basis
import numpy as np
import numpy.testing as npt



#================================================================================
#
# Class definitions
#
#================================================================================
class BasisTestCase(unittest.TestCase):
        """Tests for `basis.py`."""

        #================================================================================
        # eval_basis
        def test_eval_basis_gauss(self):
            """Are the basis correctly evaluated at the Gaussian quadrature nodes?"""
            order = 3
            test_basis = basis.Basis(order)
            phi,dphi_w = test_basis.eval_basis_gauss(order)
            npt.assert_array_almost_equal(phi, np.array([[ 1.0, -7.74596669e-01,  0.4, 0.0],
                                                         [ 1.0,  0.0           , -0.5, 0.0],
                                                         [ 1.0,  7.74596669e-01,  0.4, 0.0]]), decimal = 9)

            npt.assert_array_almost_equal(dphi_w, np.array([[ 0., 5./9., -1.29099445,  5./3.],
                                                            [ 0., 8./9.,  0.        , -4./3.],
                                                            [ 0., 5./9.,  1.29099445,  5./3.]]), decimal = 9)


        #================================================================================
        # mass_matrix
        def test_mass_matrix(self):
            """Is the mass matrix and its inverse correct?"""
            order = 3
            test_basis = basis.Basis(order)
            m,minv = test_basis.mass_matrix(order)
            npt.assert_array_almost_equal(m, [1,1.0/3.0,1.0/5.0,1.0/7.0], decimal = 9)
            npt.assert_array_almost_equal(minv, [1,3,5,7], decimal = 9)

            
if __name__ == '__main__':
    unittest.main()
                
