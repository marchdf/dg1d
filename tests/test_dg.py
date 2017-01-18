#=========================================================================
#
# Imports
#
#=========================================================================
import unittest
from .context import solution
from .context import dg
import numpy as np
import numpy.testing as npt

#=========================================================================
#
# Class definitions
#
#=========================================================================


class DGTestCase(unittest.TestCase):
    """Tests for `dg.py`."""

    #=========================================================================
    # Set up
    def setUp(self):
        self.solution = solution.Solution('sinewave 10', 'advection', 3)
        self.dgsolver = dg.DG(self.solution)

    #=========================================================================
    # add_interior_face_fluxes
    def test_add_interior_face_fluxes(self):
        """Is the interior and face fluxes addition correct?"""

        # Modification to the data for easy testing
        self.dgsolver.F = 1.0 * np.arange(24).reshape(4, 6)
        self.dgsolver.q = 1.0 * np.arange(5)
        self.dgsolver.Q = np.zeros(
            (self.dgsolver.F.shape[0], self.dgsolver.F.shape[1] - 2))

        # Call the function that we are testing
        self.dgsolver.add_interior_face_fluxes(1)

        npt.assert_array_almost_equal(self.dgsolver.F, np.array([[0.,   0.,   1.,   2.,   3.,   5.],
                                                                 [6.,   6.,   5.,
                                                                     4.,   3.,  11.],
                                                                 [12.,  12.,  13.,
                                                                     14.,  15.,  17.],
                                                                 [18.,  18.,  17.,  16.,  15.,  23.]]), decimal=13)

    #=========================================================================
    # inverse_mass_matrix_multiply
    def test_inverse_mass_matrix_multiply(self):
        """Is the mass matrix multiplication correct?"""

        # Modification to the data for easy testing
        self.dgsolver.F = 1.0 * np.arange(25).reshape(5, 5)
        minv = np.arange(5)

        # Call the function that we are testing
        self.dgsolver.inverse_mass_matrix_multiply(minv)

        npt.assert_array_almost_equal(self.dgsolver.F, np.array([[0,  0,  0,  0,  0],
                                                                 [5,  6,  7,
                                                                     8,  9],
                                                                 [20, 22, 24,
                                                                     26, 28],
                                                                 [45, 48, 51,
                                                                     54, 57],
                                                                 [80, 84, 88, 92, 96]]), decimal=13)


if __name__ == '__main__':
    unittest.main()
