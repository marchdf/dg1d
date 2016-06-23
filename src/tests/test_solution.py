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
        # Set up
        def setUp(self):
                self.solution = solution.Solution('sinewave 10', 'advection', 3)
                self.solution.apply_ic()

        #================================================================================
        # collocate
        def test_collocate(self):
                """Is the collocation procedure correct?"""
                ug = self.solution.collocate()
                
                npt.assert_array_almost_equal(ug[:,1:-1], np.array([[ 0.17361699,  0.43839961, -0.88296246,  0.99026366, -0.7193178,  0.17361699,  0.43839961, -0.88296246,  0.99026366, -0.7193178 ],
                                                                    [ 0.7375292,  -0.19973332, -0.4143539,   0.87017202, -0.993614,   0.7375292,  -0.19973332, -0.4143539,   0.87017202, -0.993614  ],
                                                                    [ 0.993614,   -0.87017202,  0.4143539,   0.19973332, -0.7375292,  0.993614,   -0.87017202,  0.4143539,   0.19973332, -0.7375292 ],
                                                                    [ 0.7193178,  -0.99026366,  0.88296246, -0.43839961, -0.17361699, 0.7193178,  -0.99026366,  0.88296246, -0.43839961, -0.17361699]]), decimal = 7)


        #================================================================================
        # collocate_faces
        def test_collocate_faces(self):
                """Is the collocation procedure for the faces correct?"""
                uf = self.solution.collocate_faces()
                
                npt.assert_array_almost_equal(uf[:,1:-1], np.array([[-0.01874174,  0.59592943, -0.94549233,  0.93390929, -0.56560465, -0.01874174,  0.59592943, -0.94549233,  0.93390929, -0.56560465],
                                                                    [ 0.56560465, -0.93390929,  0.94549233, -0.59592943,  0.01874174,  0.56560465, -0.93390929,  0.94549233, -0.59592943,  0.01874174]]), decimal = 7)

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
