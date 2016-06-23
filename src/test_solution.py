#================================================================================
#
# Imports
#
#================================================================================
import unittest
import solution
import numpy as np
import numpy.testing as npt

#================================================================================
#
# Class definitions
#
#================================================================================
class SolutionTestCase(unittest.TestCase):
        """Tests for `solution.py`."""

        #================================================================================
        # test_ictest
        def test_ictest(self):
            """Is the initial condition setup correct (populate, x, xc, dx, ghosts, scaled_minv)?"""
            sol = solution.Solution('ictest 2', 'advection', 3)
            sol.apply_ic()

            # coefficient calculation
            npt.assert_array_almost_equal(sol.u, np.array([[ 0.,  0.11523009,  0.79166704,  0.],
                                                           [ 0.,  0.02964857,  0.33332889,  0.],
                                                           [ 0.,  0.22000253, -0.12668721,  0.],
                                                           [ 0., -0.00375639, -0.00889097,  0.]]),decimal=7)
            
            # discretization
            npt.assert_array_almost_equal(sol.x,  np.array([-1,0,1]),decimal=7)
            npt.assert_array_almost_equal(sol.xc, np.array([-0.5,0.5]),decimal=7)

            # scaled inverse mass matrix
            npt.assert_array_almost_equal(sol.scaled_minv, np.array([1./2*(2./1),3./2*(2./1),5./2*(2./1),7./2*(2./1)]),decimal=7)
            
            
if __name__ == '__main__':
    unittest.main()
