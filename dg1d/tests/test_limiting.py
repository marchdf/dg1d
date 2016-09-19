#================================================================================
#
# Imports
#
#================================================================================
import unittest
import solution
import limiting
import numpy as np
import numpy.testing as npt

#================================================================================
#
# Class definitions
#
#================================================================================
class LimitingTestCase(unittest.TestCase):
    """Tests for `limiting.py`."""


    #================================================================================
    # Set up
    def setUp(self):
        self.solution = solution.Solution('entrpyw 3', 'euler', 3, '', [-1,-1])
        self.solution.u = np.array([[0,0,0,1,1,1,2,2,2,3,3,3,0,0,0],
                                    [0,0,0,2,2,2,2,2,2,3,3,3,0,0,0],
                                    [0,0,0,1,1,1,4,4,4,3,3,3,0,0,0],
                                    [0,0,0,1,1,1,2,2,2,3,3,3,0,0,0]],dtype=float)
        self.solution.apply_bc()
        self.biswas_limiter  = limiting.Limiter('full_biswas',self.solution)
        self.hr_limiter      = limiting.Limiter('adaptive_hr',self.solution)
        
    #================================================================================
    # test_biswas_limiting_procedure
    def test_biswas_limiting_procedure(self):
        """Is the Biswas limiting procedure correct?"""

        # Test limiting
        self.biswas_limiter.limit(self.solution)
        npt.assert_array_almost_equal(self.solution.u, np.array([[ 3.,  3.,  3.,  1.,  1.,  1.,  2.,  2.,  2.,  3.,  3.,  3.,  1.,  1.,  1.,],
                                                                 [ 3.,  3.,  3.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  2.,  2.,  2.,],
                                                                 [ 3.,  3.,  3.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,],
                                                                 [ 3.,  3.,  3.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,]]))
        

    #================================================================================
    # test_deltas
    def test_deltas(self):
        """Is the calculation for the deltas is correct?"""

        dm, dp = self.biswas_limiter.deltas(self.solution)


        # Make sure they are correct
        npt.assert_array_almost_equal(dm, np.array([[-2.,         -2.,         -2.,          1.,  1.,  1.,  1.,          1.,          1.        ],
                                                    [-0.33333333, -0.33333333, -0.33333333,  0.,  0.,  0.,  0.33333333,  0.33333333,  0.33333333],
                                                    [-0.4,        -0.4,        -0.4,         0.6, 0.6, 0.6, -0.2,       -0.2,        -0.2       ]]))

        npt.assert_array_almost_equal(dp, np.array([[ 1.,  1.,  1.,   1.,          1.,          1.,          -2.,         -2.,         -2.        ],
                                                    [ 0.,  0.,  0.,   0.33333333,  0.33333333,  0.33333333,  -0.33333333, -0.33333333, -0.33333333],
                                                    [ 0.6, 0.6, 0.6, -0.2,        -0.2,        -0.2,         -0.4,        -0.4,        -0.4       ]]))


    #================================================================================
    # test_minmod
    def test_minmod(self):
        """Is the minmod procedure correct?"""

        dummy_solution = solution.Solution('sinewave 3', 'advection', 3)
        limiter  = limiting.Limiter('full',dummy_solution)

        # Dummy variables
        A = np.array([[0,0,-1],
                      [3,-4,11]])
        B = np.array([[-2,1,2],
                      [-3,-8,10]])
        C = np.array([[-3,3,5],
                      [4,-1,6]])

        # Minmod procedure
        M = limiter.minmod(A,B,C)
        npt.assert_array_almost_equal(M, np.array([[0,0,0],
                                                   [0,-1,6]]),decimal=7)


    #================================================================================
    # test_hr_limiting_procedure
    def test_hr_limiting_procedure(self):
        """Is the adaptive HR limiting procedure correct?"""

        # Do the sensors. In this case we are limiting everywhere
        self.solution.sensors.sensing(self.solution)
        
        # Test limiting
        self.hr_limiter.limit(self.solution)
        # npt.assert_array_almost_equal(self.solution.u, np.array([[ 3.,  3.,  3.,  1.,  1.,  1.,  2.,  2.,  2.,  3.,  3.,  3.,  1.,  1.,  1.,],
        #                                                          [ 3.,  3.,  3.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  2.,  2.,  2.,],
        #                                                          [ 3.,  3.,  3.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,],
        #                                                          [ 3.,  3.,  3.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,]]))



if __name__ == '__main__':
    unittest.main()
