#================================================================================
#
# Imports
#
#================================================================================
import unittest
import euler_fluxes
import numpy as np
import numpy.testing as npt

#================================================================================
#
# Class definitions
#
#================================================================================
class EulerFluxesTestCase(unittest.TestCase):
        """Tests for `euler_fluxes.py`."""

        #================================================================================
        # max_wave_speed
        def test_max_wave_speed(self):
            """Is the max wave speed solver correct?"""

            # toy data
            u = np.arange(1,12*3+1).reshape((3,12))

            # Get the maximum wave speed
            m = euler_fluxes.max_wave_speed(u)

            # test
            npt.assert_array_almost_equal(m,2.74833147735, decimal = 7)


        #================================================================================
        # riemann_rusanov
        def test_riemann_rusanov(self):
            """Is the Rusanov Riemann solver correct?"""

            # Left/right toy data
            ul = np.arange(1,13)
            ur = np.arange(1,13)[::-1]

            # Get the flux
            F = euler_fluxes.riemann_rusanov(ul,ur)

            # test
            npt.assert_array_almost_equal(F,
                                          np.array([ -8.61582313, -4.13415831, -0.72679906, 1.78892781, 5.11780113, 7.24999235, 7.3690381, 10.53092381, 12.48640363, 12.37032176, 15.52088988,  17.51156911]),
                                          decimal = 7)
            
        #================================================================================
        # interior_flux
        def test_interior_flux(self):
            """Is the interior flux correct?"""

            # toy data
            u = np.arange(1,12*3+1).reshape((3,12))

            # Get the maximum wave speed
            F = euler_fluxes.interior_flux(u)
            
            # test
            npt.assert_array_almost_equal(F,
                                          np.array([[ 2., 4.4, 6.8, 5., 7.4, 8.9375, 8., 10.91428571, 12.31020408, 11., 14.48, 15.818],
                                                    [ 14., 18.06153846, 19.36804734, 17., 21.65, 22.93671875, 20., 25.24210526, 26.51523546, 23., 28.83636364, 30.09958678],
                                                    [ 26., 32.432, 33.68768, 29., 36.02857143, 37.27831633, 32., 39.62580645, 40.87075963, 35., 43.22352941, 44.46453287]]),
                                          decimal = 7)
            

if __name__ == '__main__':
        unittest.main()

