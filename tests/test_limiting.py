# =========================================================================
#
# Imports
#
# =========================================================================
import unittest
from .context import solution
from .context import limiting
import numpy as np
import numpy.testing as npt


# =========================================================================
#
# Class definitions
#
# =========================================================================
class LimitingTestCase(unittest.TestCase):
    """Tests for `limiting.py`."""

    # =========================================================================
    def setUp(self):

        # p = 1 solution
        self.hr1 = solution.Solution(
            'entrpyw 5', 'euler', 1, '', '', [-1, -1])
        self.hr1.apply_bc()
        self.hr1_limiter = limiting.Limiter('adaptive_hr', self.hr1)

        # p = 2 solution
        self.hr2 = solution.Solution(
            'entrpyw 5', 'euler', 2, '', '', [-1, -1])
        self.hr2.apply_bc()
        self.hr2_limiter = limiting.Limiter('adaptive_hr', self.hr2)

        # p = 3 solution
        self.hr3 = solution.Solution(
            'entrpyw 5', 'euler', 3, '', '', [-1, -1])
        self.hr3.apply_bc()
        self.hr3_limiter = limiting.Limiter('adaptive_hr', self.hr3)

    # =========================================================================
    def test_hr1_limiting_procedure(self):
        """Is the adaptive HR limiting for p = 1 procedure correct?"""

        # Do the sensors. In this case we are limiting everywhere
        self.hr1.sensors.sensing(self.hr1)

        # Test limiting
        self.hr1_limiter.limit(self.hr1)
        npt.assert_array_almost_equal(self.hr1.u,
                                      np.array(
                                          [[1.10990656,  1.10990656,  3.05495328,  0.89009344,  0.89009344,  2.94504672,
                                            0.82216745,  0.82216745,  2.91108373,  1., 1., 3.,
                                            1.17783255,  1.17783255,  3.08891627,  1.10990656,  1.10990656,  3.05495328,
                                            0.89009344,  0.89009344,  2.94504672],
                                           [-0.03396299, - 0.03396299, -0.0169815, -0.03396299, -0.03396299, -0.0169815,
                                            0., 0., 0., 0.08891627, 0.08891627,  0.04445814,
                                            0., 0., 0., -0.03396299, -0.03396299, -0.0169815,
                                            -0.03396299, -0.03396299, -0.0169815]]))

    # ======================================================================
    def test_hr2_limiting_procedure(self):
        """Is the adaptive HR limiting for p = 2 procedure correct?"""

        # Do the sensors. In this case we are limiting everywhere
        self.hr2.sensors.sensing(self.hr2)

        # Test limiting
        self.hr2_limiter.limit(self.hr2)
        npt.assert_array_almost_equal(self.hr2.u,
                                      np.array(
                                          [[1.10997359e+00,  1.10997359e+00,  3.05498679e+00,  8.90026412e-01,
                                            8.90026412e-01,  2.94501321e+00,  8.22058997e-01,  8.22058997e-01,
                                            2.91102950e+00,  1.00000000e+00,  1.00000000e+00,  3.00000000e+00,
                                            1.17794100e+00,   1.17794100e+00,   3.08897050e+00,   1.10997359e+00,
                                            1.10997359e+00,   3.05498679e+00,   8.90026412e-01,   8.90026412e-01,
                                            2.94501321e+00],
                                           [-3.39837076e-02, -3.39837076e-02, -1.69918538e-02, -3.39837076e-02,
                                            -3.39837076e-02, -1.69918538e-02,   7.73823581e-03,   7.73823581e-03,
                                            3.86911791e-03,   8.89705015e-02,   8.89705015e-02,   4.44852507e-02,
                                            7.73823581e-03,   7.73823581e-03,   3.86911791e-03, -3.39837076e-02,
                                            -3.39837076e-02, -1.69918538e-02, -3.39837076e-02, -3.39837076e-02,
                                            -1.69918538e-02],
                                           [-1.38777878e-17, -1.38777878e-17, -1.22587126e-16,   0.00000000e+00,
                                            0.00000000e+00,   0.00000000e+00,   1.39073145e-02,   1.39073145e-02,
                                            6.95365723e-03,   0.00000000e+00,  0.00000000e+00,   0.00000000e+00,
                                            -1.39073145e-02, -1.39073145e-02, -6.95365723e-03, -1.38777878e-17,
                                            -1.38777878e-17, -1.22587126e-16,  0.00000000e+00,   0.00000000e+00,
                                            0.00000000e+00]]))

    # ======================================================================
    def test_hr3_limiting_procedure(self):
        """Is the adaptive HR limiting for p = 3 procedure correct?"""

        # Do the sensors. In this case we are limiting everywhere
        self.hr3.sensors.sensing(self.hr3)

        # Test limiting
        self.hr3_limiter.limit(self.hr3)
        npt.assert_array_almost_equal(self.hr3.u,
                                      np.array(
                                          [[1.10997336e+00,   1.10997336e+00,   3.05498668e+00,   8.90026639e-01,
                                            8.90026639e-01,   2.94501332e+00,   8.22059365e-01,   8.22059365e-01,
                                            2.91102968e+00,   1.00000000e+00,   1.00000000e+00,   3.00000000e+00,
                                            1.17794064e+00,   1.17794064e+00,   3.08897032e+00,   1.10997336e+00,
                                            1.10997336e+00,   3.05498668e+00,   8.90026639e-01,   8.90026639e-01,
                                            2.94501332e+00],
                                           [-5.81479895e-02, -5.81479895e-02, -2.90739948e-02, -5.81479895e-02,
                                            -5.81479895e-02, -2.90739948e-02,   6.61865938e-03,   6.61865938e-03,
                                            3.30932969e-03,  1.15735473e-01,   1.15735473e-01,   5.78677367e-02,
                                            6.61865938e-03,  6.61865938e-03,   3.30932969e-03,  -5.81479895e-02,
                                            -5.81479895e-02, -2.90739948e-02, -5.81479895e-02, -5.81479895e-02,
                                            -2.90739948e-02],
                                           [-4.64699081e-03, -4.64699081e-03, -2.32349540e-03,  4.64699081e-03,
                                            4.64699081e-03,  2.32349540e-03, 1.35340989e-02,  1.35340989e-02,
                                            6.76704945e-03,  6.49084445e-17, 6.49084445e-17,  2.17491393e-16,
                                            -1.35340989e-02,  -1.35340989e-02, -6.76704945e-03,  -4.64699081e-03,
                                            -4.64699081e-03,  -2.32349540e-03,  4.64699081e-03,  4.64699081e-03,
                                            2.32349540e-03],
                                           [9.29398162e-04, 9.29398162e-04, 4.64699081e-04, 9.29398162e-04,
                                            9.29398162e-04, 4.64699081e-04, -1.56761345e-18, -1.56761345e-18,
                                            -7.83806724e-19, -2.43319598e-03, -2.43319598e-03, -1.21659799e-03,
                                            -1.56761345e-18, -1.56761345e-18, -7.83806724e-19,  9.29398162e-04,
                                            9.29398162e-04, 4.64699081e-04, 9.29398162e-04, 9.29398162e-04,
                                            4.64699081e-04]]))

    # =========================================================================
    def test_legendre_to_monomial(self):
        """Is the Legendre to monomial procedure correct?"""

        u = np.array([0., 1., 2., 3.])
        p = self.hr3_limiter.legendre_to_monomial(u)

        # Test
        npt.assert_array_almost_equal(p, np.array(
            [-1. * 1, -3.5 * 1, 3. * 2, 7.5 * 6]))

    # =========================================================================
    def test_monomial_to_legendre(self):
        """Is the monomial to Legendre procedure correct?"""

        p = np.array([-1. * 1, -3.5 * 1, 3. * 2, 7.5 * 6])
        u = self.hr3_limiter.monomial_to_legendre(p)

        # Test
        npt.assert_array_almost_equal(u, np.array([0., 1., 2., 3.]))

    # =========================================================================
    def test_limit_monomial_p1(self):
        """Is the monomial limiting procedure correct for a linear polynomial?

        The only coefficient that matters here is the zeroth order
        one. Its really a test of the minmod procedure.
        """

        # Linear polynomial
        alim = self.hr1_limiter.limit_monomial(np.array([3., 4]),  # center
                                               np.array([1., 2]),  # left
                                               np.array([5., 6]))  # right
        npt.assert_array_almost_equal(alim, np.array([3., 1]))

        # Another linear polynomial
        alim = self.hr1_limiter.limit_monomial(np.array([-1., 3]),
                                               np.array([0., -3]),
                                               np.array([2., 1]))
        npt.assert_array_almost_equal(alim, np.array([-1., 0]))

        # Another linear polynomial
        alim = self.hr1_limiter.limit_monomial(np.array([-3., 3]),
                                               np.array([-1., -3]),
                                               np.array([-4., 1]))
        npt.assert_array_almost_equal(alim, np.array([-3., -0.5]))

    # =========================================================================
    def test_limit_monomial_p2(self):
        """Is the monomial limiting procedure correct for a quadratic polynomial?
        """
        alim = self.hr2_limiter.limit_monomial(np.array([-1., 5, 2]),
                                               np.array([1., 2, 3]),
                                               np.array([0., 6, 3]))
        npt.assert_array_almost_equal(alim, np.array([-0.75, 0, 0.5]))

        alim = self.hr2_limiter.limit_monomial(np.array([1., 2, 3]),
                                               np.array([0., 6, 3]),
                                               np.array([-1., 5, 2]))
        npt.assert_array_almost_equal(alim, np.array([1.5, 0.0, 0.0]))

    # =========================================================================
    def test_limit_monomial_p3(self):
        """Is the monomial limiting procedure correct for a cubic polynomial?
        """
        alim = self.hr3_limiter.limit_monomial(np.array([-1., 5, 2, 7]),
                                               np.array([1., 2, 3, 1]),
                                               np.array([0., 6, 1, 2]))
        npt.assert_array_almost_equal(alim, np.array(
            [-0.76388889, 0., 0.58333333, -0.5]))

    # =========================================================================
    def test_integrate_monomial_derivative(self):
        """Is the integration of monomial derivative correct?"""
        res1 = [limiting.integrate_monomial_derivative(
            1, n) for n in range(1, 6)]
        res2 = [limiting.integrate_monomial_derivative(
            2, n) for n in range(2, 6)]

        # Test
        npt.assert_equal(res1, [2.0, 0.0, 1. / 3, 0.0, 1. / 60])
        npt.assert_equal(res2, [2.0, 0.0, 1. / 3, 0.0])

    # =========================================================================
    def test_integrate_monomial_derivative_bounds(self):
        """Is the integration of monomial derivative with bounds correct?"""
        res1 = [limiting.integrate_monomial_derivative_bounds(
            1, n, -3, -1) for n in range(1, 6)]
        res2 = [limiting.integrate_monomial_derivative_bounds(
            2, n, 1, 3) for n in range(2, 6)]

        # Test
        npt.assert_equal(res1, [2., -4., 13. / 3, -10. / 3, 121. / 60])
        npt.assert_equal(res2, [2., 4, 13. / 3, 10. / 3])

    # =========================================================================
    def test_scalar_minmod(self):
        """Is the scalar minmod function correct?"""
        npt.assert_equal(limiting.scalar_minmod(-1, 0.5), 0)
        npt.assert_equal(limiting.scalar_minmod(0.7, 0.5), 0.5)
        npt.assert_equal(limiting.scalar_minmod(0.7, -0.5), 0)
        npt.assert_equal(limiting.scalar_minmod(-0.7, -0.2), -0.2)


if __name__ == '__main__':
    unittest.main()
