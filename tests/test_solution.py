# =========================================================================
#
# Imports
#
# =========================================================================
import unittest
import dg1d.solution as solution
import numpy as np
import numpy.testing as npt

# =========================================================================
#
# Class definitions
#
# =========================================================================


class SolutionTestCase(unittest.TestCase):
    """Tests for `solution.py`."""

    # =========================================================================
    # Set up
    def setUp(self):
        self.solution = solution.Solution('sinewave 10', 'advection', 3)

    # =========================================================================
    # collocate
    def test_collocate(self):
        """Is the collocation procedure correct?"""
        ug = self.solution.collocate()

        npt.assert_array_almost_equal(ug[:, 1:-1],
                                      np.array([[0.08713997,  0.9743665,  0.51505165, -0.65604708, -0.92051104, 0.08713997,  0.9743665,  0.51505165, -0.65604708, -0.92051104],
                                                [0.40291734,  0.99494982,  0.21199547, -0.86392942, -0.74593321,
                                                    0.40291734,  0.99494982,  0.21199547, -0.86392942, -0.74593321],
                                                [0.74593321,  0.86392942, -0.21199547, -0.99494982, -0.40291734,
                                                    0.74593321,  0.86392942, -0.21199547, -0.99494982, -0.40291734],
                                                [0.92051104,  0.65604708, -0.51505165, -0.9743665, -0.08713997, 0.92051104,  0.65604708, -0.51505165, -0.9743665, -0.08713997]]),
                                      decimal=7)

    # =========================================================================
    # collocate_faces
    def test_collocate_faces(self):
        """Is the collocation procedure for the faces correct?"""

        uf = self.solution.collocate_faces()

        npt.assert_array_almost_equal(uf[:, 1:-1],
                                      np.array([[-7.03058141e-04,  9.49622368e-01,  5.87601958e-01, -5.86464386e-01, -9.50056882e-01, -7.03058141e-04,  9.49622368e-01,  5.87601958e-01, -5.86464386e-01, -9.50056882e-01],
                                                [9.50056882e-01,  5.86464386e-01, -5.87601958e-01, -9.49622368e-01,  7.03058141e-04,  9.50056882e-01,  5.86464386e-01, -5.87601958e-01, -9.49622368e-01,  7.03058141e-04]]),
                                      decimal=7)

    # =========================================================================
    # test_ictest
    def test_ictest(self):
        """Is the initial condition setup correct?"""
        sol = solution.Solution('ictest 2', 'advection', 3)

        # coefficient calculation
        npt.assert_array_almost_equal(sol.u, np.array([[0.,  0.11523009,  0.79166704,  0.],
                                                       [0.,  0.02964857,
                                                           0.33332889,  0.],
                                                       [0.,  0.22000253, -
                                                           0.12668721,  0.],
                                                       [0., -0.00375639, -0.00889097,  0.]]), decimal=7)

        # discretization
        npt.assert_array_almost_equal(sol.x,  np.array([-1, 0, 1]), decimal=7)
        npt.assert_array_almost_equal(sol.xc, np.array([-0.5, 0.5]), decimal=7)

        # scaled inverse mass matrix
        npt.assert_array_almost_equal(sol.scaled_minv, np.array(
            [1. / 2 * (2. / 1), 3. / 2 * (2. / 1), 5. / 2 * (2. / 1), 7. / 2 * (2. / 1)]), decimal=7)


if __name__ == '__main__':
    unittest.main()
