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
        self.solution_biswas = solution.Solution('entrpyw 3', 'euler', 3, '', [-1,-1])
        self.solution_biswas.u = np.array([[0,0,0,1,1,1,2,2,2,3,3,3,0,0,0],
                                           [0,0,0,2,2,2,2,2,2,3,3,3,0,0,0],
                                           [0,0,0,1,1,1,4,4,4,3,3,3,0,0,0],
                                           [0,0,0,1,1,1,2,2,2,3,3,3,0,0,0]],dtype=float)
        self.solution_biswas.apply_bc()
        self.biswas_limiter  = limiting.Limiter('full_biswas',self.solution_biswas)



        self.solution_hr = solution.Solution('entrpyw 3', 'euler', 3, '', [-1,-1])
        self.solution_hr.u = np.array([[0,0,0,1,1,1,2,2,2,3,3,3,0,0,0],
                                       [0,0,0,2,2,2,2,4,2,9,8,3,0,0,0],
                                       [0,0,0,3,1,4,6,3,4,1,6,3,0,0,0],
                                       [0,0,0,4,1,5,7,2,2,2,4,3,0,0,0]],dtype=float)
        self.solution_hr.apply_bc()
        self.hr_limiter      = limiting.Limiter('adaptive_hr',self.solution_hr)
        

    #================================================================================
    # test_biswas_limiting_procedure
    def test_biswas_limiting_procedure(self):
        """Is the Biswas limiting procedure correct?"""

        # Test limiting
        self.biswas_limiter.limit(self.solution_biswas)
        npt.assert_array_almost_equal(self.solution_biswas.u, np.array([[ 3.,  3.,  3.,  1.,  1.,  1.,  2.,  2.,  2.,  3.,  3.,  3.,  1.,  1.,  1.,],
                                                                        [ 3.,  3.,  3.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  2.,  2.,  2.,],
                                                                        [ 3.,  3.,  3.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,],
                                                                        [ 3.,  3.,  3.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,]]))
        

    #================================================================================
    # test_deltas
    def test_deltas(self):
        """Is the calculation for the deltas is correct?"""

        dm, dp = self.biswas_limiter.deltas(self.solution_biswas)

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
        self.solution_hr.sensors.sensing(self.solution_hr)

        # Test limiting
        self.hr_limiter.limit(self.solution_hr)
        npt.assert_array_almost_equal(self.solution_hr.u, np.array([[ 3., 3., 3.,  1.,  1., 1., 2.,          2.,  2.,          3.,          3.,  3.,  1., 1., 1.],
                                                                    [ 9., 8., 3., -1.7, 0., 0., 0.,         -2.8, 0.,         -0.23333333,  0., -0.6, 2., 2., 2.],
                                                                    [ 1., 6., 3.,  0.,  0., 0., 0.33333333,  0.,  0.33333333,  0.,          0.,  0.,  3., 1., 4.],
                                                                    [ 2., 4., 3.,  0.2, 0., 0., 0.,          0.3, 0.,          0.06666667,  0.,  0.1, 4., 1., 5.]]))

    #================================================================================
    # test_legendre_to_monomial
    def test_legendre_to_monomial(self):
        """Is the Legendre to monomial procedure correct?"""

        u = np.array([0.,1.,2.,3.])
        p = self.hr_limiter.legendre_to_monomial(u)
        
        # Test
        npt.assert_array_almost_equal(p, np.array([-1.*1, -3.5*1, 3.*2, 7.5*6]))


    #================================================================================
    # test_monomial_to_legendre
    def test_monomial_to_legendre(self):
        """Is the monomial to Legendre procedure correct?"""

        p = np.array([-1.*1, -3.5*1, 3.*2, 7.5*6])
        u = self.hr_limiter.monomial_to_legendre(p)
        
        # Test
        npt.assert_array_almost_equal(u,np.array([0.,1.,2.,3.]))

    #================================================================================
    # test_limit_monomial_linear
    def test_limit_monomial_linear(self):
        """Is the monomial limiting procedure correct for a linear polynomial?

        The only coefficient that matters here is the zeroth order
        one. Its really a test of the minmod procedure.
        """

        # Linear polynomial
        alim = self.hr_limiter.limit_monomial(np.array([3.,4]), # center
                                              np.array([1.,2]), # left
                                              np.array([5.,6])) # right
        npt.assert_array_almost_equal(alim,np.array([3.,1]))

        # Another linear polynomial
        alim = self.hr_limiter.limit_monomial(np.array([-1.,3]), # center
                                              np.array([0.,-3]), # left
                                              np.array([2.,1]))  # right
        npt.assert_array_almost_equal(alim,np.array([-1.,0]))

        # Another linear polynomial
        alim = self.hr_limiter.limit_monomial(np.array([-3.,3]),  # center
                                              np.array([-1.,-3]), # left
                                              np.array([-4.,1]))  # right
        npt.assert_array_almost_equal(alim,np.array([-3.,-0.5]))

    #================================================================================
    # test_limit_monomial
    def test_limit_monomial(self):
        """Is the monomial limiting procedure correct for a general polynomial?
        """

        # Quadratic polynomial
        alim = self.hr_limiter.limit_monomial(np.array([-1.,5,2]), # center
                                              np.array([1.,2,3]),  # left
                                              np.array([0.,6,3]))  # right
        npt.assert_array_almost_equal(alim,np.array([-0.75,0,0.5]))

        # Cubic polynomial
        alim = self.hr_limiter.limit_monomial(np.array([-1.,5,2,7]), # center
                                              np.array([1.,2,3,1]),  # left
                                              np.array([0.,6,1,2]))  # right
        npt.assert_array_almost_equal(alim,np.array([-0.76388889,0.,0.58333333, -0.5]))
        
    #================================================================================
    # test_integrate_monomial_derivative
    def test_integrate_monomial_derivative(self):
        """Is the integration of monomial derivative correct?"""
        res1 = [self.hr_limiter.integrate_monomial_derivative(1,n) for n in range(1,6)]
        res2 = [self.hr_limiter.integrate_monomial_derivative(2,n) for n in range(2,6)] 

        # Test
        npt.assert_equal(res1,[2.0,0.0,1./3,0.0,1./60])
        npt.assert_equal(res2,[2.0,0.0,1./3,0.0])
        

    #================================================================================
    # test_integrate_monomial_derivative_bounds
    def test_integrate_monomial_derivative_bounds(self):
        """Is the integration of monomial derivative with bounds correct?"""
        res1 = [self.hr_limiter.integrate_monomial_derivative_bounds(1,n,-3,-1) for n in range(1,6)]
        res2 = [self.hr_limiter.integrate_monomial_derivative_bounds(2,n,1,3) for n in range(2,6)] 

        # Test
        npt.assert_equal(res1,[2.,-4.,13./3,-10./3,121./60])
        npt.assert_equal(res2,[2.,4,13./3,10./3])

    #================================================================================
    # test_scalar_minmod
    def test_scalar_minmod(self):
        """Is the scalar minmod function correct?"""
        npt.assert_equal(self.hr_limiter.scalar_minmod(-1,0.5), 0)
        npt.assert_equal(self.hr_limiter.scalar_minmod(0.7,0.5), 0.5)
        npt.assert_equal(self.hr_limiter.scalar_minmod(0.7,-0.5), 0)
        npt.assert_equal(self.hr_limiter.scalar_minmod(-0.7,-0.2), -0.2)


if __name__ == '__main__':
    unittest.main()
