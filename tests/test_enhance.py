#=========================================================================
#
# Imports
#
#=========================================================================
import unittest
from .context import enhance
from .context import basis
import numpy as np
import numpy.testing as npt

#=========================================================================
#
# Class definitions
#
#=========================================================================


class EnhanceTestCase(unittest.TestCase):
    """Tests for `enhance.py`."""

    #=========================================================================
    # enhancement_matrix
    def test_enhancement_matrices(self):
        """Are the enhancement matrices correct?"""

        # Original solution order
        solution_order = 1

        # Modes we want to use from the neighbor
        modes = [1]

        # Get the enhancement matrix
        Al, Alinv, Ar, Arinv = enhance.enhancement_matrices(
            solution_order, modes)

        # Make sure both left and right enhancements are correct
        npt.assert_array_almost_equal(Al, np.array([[1.,  0.,  0.],
                                                    [0.,  1.,  0.],
                                                    [0.,  1.,  6.]]), decimal=13)

        npt.assert_array_almost_equal(Ar, np.array([[1.,  0.,  0.],
                                                    [0.,  1.,  0.],
                                                    [0.,  1., -6.]]), decimal=13)

    #=========================================================================
    # enhancement_vectors
    def test_left_enhancement_vectors(self):
        """Are the left enhancement vectors correct?"""

        # Original solution order
        solution_order = 2

        # Modes we want to use from the neighbor
        modes = [0, 1]

        # Generate the basis
        order = solution_order + len(modes)
        base = basis.Basis(order)

        # Get the enhancement matrices
        Al, Alinv, Ar, Arinv = enhance.enhancement_matrices(
            solution_order, modes)

        # Get the enhancement vectors
        alphaL, alphaR, betaL, betaR = enhance.left_enhancement_vectors(
            Alinv, Arinv, solution_order, modes, base.psi)

        # Make sure both left and right enhancement vectors are correct
        npt.assert_array_almost_equal(alphaL, np.array(
            [0.7734375,  0.6796875,  0.4375]), decimal=7)
        npt.assert_array_almost_equal(alphaR, np.array(
            [0.2265625, -0.1328125,  0.]), decimal=7)
        npt.assert_array_almost_equal(betaL, np.array(
            [0.2265625,  0.1328125,  0.]), decimal=7)
        npt.assert_array_almost_equal(betaR, np.array(
            [0.7734375, -0.6796875,  0.4375]), decimal=7)

    #=========================================================================
    # face_value
    def test_face_value(self):
        """Is the enhanced evaluation of the faces, given a solution, correct?"""

        # Setup the procedure
        order = 2

        # Generate a dummy solution
        u = np.arange(1, (order + 1) * 5 +
                      1).reshape((order + 1, 5), order='F')
        enhancement_type = 'icb 0 1'
        enhanced = enhance.Enhance(order, enhancement_type, u.shape[1])

        # Calculate the enhance face values
        uf = enhanced.face_value(u, 1)

        # Test
        npt.assert_array_almost_equal(uf, np.array([[0.,     2.8125,    5.484375,  8.15625,  10.828125],
                                                    [3.6875, 9.640625, 15.59375,  21.546875,  0.]]), decimal=7)


if __name__ == '__main__':
    unittest.main()
