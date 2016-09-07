#================================================================================
#
# Imports
#
#================================================================================
import unittest
import solution
import sensor
import numpy as np
import numpy.testing as npt

#================================================================================
#
# Class definitions
#
#================================================================================
class SensorTestCase(unittest.TestCase):
        """Tests for `sensors.py`."""

        #================================================================================
        # advection sensing
        def test_sensing_advection(self):
                """Is the sensing procedure for advection correct?"""

                # Initialize
                sol = solution.Solution('sinewave 10', 'advection', 3)
                sen = sensor.Sensor([0.46],sol)

                # Try the sensors
                sen.sensing(sol)

                # test
                npt.assert_array_equal(sen.sensors,
                                       np.array([0,0,1,1,1,1,1,1,1,1,0,0]))


        #================================================================================
        # advection sensing
        def test_sensing_euler(self):
                """Is the sensing procedure for Euler equations correct?"""

                # Initialize
                sol = solution.Solution('entrpyw 3', 'euler', 3)
                sen = sensor.Sensor([0.46,0.4],sol)

                # Try the sensors
                sen.sensing(sol)

                # test
                npt.assert_array_equal(sen.sensors,
                                       np.array([0,0,1,1,1,1,1,1,1,1,0,0]))








            
if __name__ == '__main__':
        unittest.main()
