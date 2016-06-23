#================================================================================
#
# Imports
#
#================================================================================
import unittest
import solution
import dg
import numpy as np
import numpy.testing as npt

#================================================================================
#
# Class definitions
#
#================================================================================
class DGTestCase(unittest.TestCase):
        """Tests for `dg.py`."""
        
        #================================================================================
        # collocate
        def test_collocate(self):
            """Is the collocation procedure correct?"""
            sol = solution.Solution('sinewave 10', 'advection', 3)
            sol.apply_ic()
            ug = dg.collocate(sol)

            npt.assert_array_almost_equal(ug[:,1:-1], np.array([[ 0.17361699,  0.43839961, -0.88296246,  0.99026366, -0.7193178,  0.17361699,  0.43839961, -0.88296246,  0.99026366, -0.7193178 ],
                                                        [ 0.7375292,  -0.19973332, -0.4143539,   0.87017202, -0.993614,   0.7375292,  -0.19973332, -0.4143539,   0.87017202, -0.993614  ],
                                                        [ 0.993614,   -0.87017202,  0.4143539,   0.19973332, -0.7375292,  0.993614,   -0.87017202,  0.4143539,   0.19973332, -0.7375292 ],
                                                        [ 0.7193178,  -0.99026366,  0.88296246, -0.43839961, -0.17361699, 0.7193178,  -0.99026366,  0.88296246, -0.43839961, -0.17361699]]), decimal = 7)


        #================================================================================
        # collocate_faces
        def test_collocate_faces(self):
            """Is the collocation procedure for the faces correct?"""
            sol = solution.Solution('sinewave 10', 'advection', 3)
            sol.apply_ic()
            uf = dg.collocate_faces(sol)
            
            npt.assert_array_almost_equal(uf[:,1:-1], np.array([[-0.01874174,  0.59592943, -0.94549233,  0.93390929, -0.56560465, -0.01874174,  0.59592943, -0.94549233,  0.93390929, -0.56560465],
                                                        [ 0.56560465, -0.93390929,  0.94549233, -0.59592943,  0.01874174,  0.56560465, -0.93390929,  0.94549233, -0.59592943,  0.01874174]]), decimal = 7)


        #================================================================================
        # inverse_mass_matrix_multiply
        def test_inverse_mass_matrix_multiply(self):
            """Is the mass matrix multiplication correct?"""
            a = np.arange(25).reshape(5,5)
            b = np.arange(5)
            res = dg.inverse_mass_matrix_multiply(a,b)
            npt.assert_array_almost_equal(res, np.array([[ 0,  0,  0,  0,  0],
                                                         [ 5,  6,  7,  8,  9],
                                                         [20, 22, 24, 26, 28],
                                                         [45, 48, 51, 54, 57],
                                                         [80, 84, 88, 92, 96]]),decimal=7)

        #================================================================================
        # add_interior_face_fluxes
        def test_add_interior_face_fluxes(self):
            """Is the interior and face fluxes addition correct?"""

            F = 1.0*np.arange(24).reshape(4,6)
            q = 1.0*np.arange(5)
            dg.add_interior_face_fluxes(F,q)
            
            npt.assert_array_almost_equal(F, np.array([[  0.,   0.,   1.,   2.,   3.,   5.],
                                                       [  6.,   6.,   5.,   4.,   3.,  11.],
                                                       [ 12.,  12.,  13.,  14.,  15.,  17.],
                                                       [ 18.,  18.,  17.,  16.,  15.,  23.]]),decimal=7)

            
if __name__ == '__main__':
    unittest.main()
                
