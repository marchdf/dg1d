#================================================================================
#
# Imports
#
#================================================================================
import unittest
import riemann
import numpy as np
import numpy.testing as npt

#================================================================================
#
# Class definitions
#
#================================================================================
class RiemannTestCase(unittest.TestCase):
        """Tests for `riemann.py`."""

        #================================================================================
        # euler_rusanov
        def test_euler_rusanov(self):
            """Is the Rusanov Riemann solver correct?"""

            # Left/right data
            ul = np.arange(1,13)
            ur = np.arange(1,13)[::-1]

            # Get the flux
            F = riemann.euler_rusanov(ul,ur)

            # test
            npt.assert_array_almost_equal(F,
                                          np.array([ -8.61582313, -4.13415831, -0.72679906, 1.78892781, 5.11780113, 7.24999235, 7.3690381, 10.53092381, 12.48640363, 12.37032176, 15.52088988,  17.51156911]),
                                          decimal = 7)
            



if __name__ == '__main__':
        unittest.main()

