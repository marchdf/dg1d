#=========================================================================
#
# Imports
#
#=========================================================================
import unittest
import dg1d.solution as solution
import dg1d.sensor as sensor
import numpy as np
import numpy.testing as npt

#=========================================================================
#
# Class definitions
#
#=========================================================================


class SensorTestCase(unittest.TestCase):
    """Tests for `sensors.py`."""

    #=========================================================================
    # advection sensing
    def test_sensing_advection(self):
        """Is the sensing procedure for advection correct?"""

        # Initialize
        sol = solution.Solution('sinewave 10', 'advection', 3)
        sen = sensor.Sensor([0.46], sol.N_E + 2)

        # Try the sensors
        sen.sensing(sol)

        # test
        npt.assert_array_equal(sen.sensors,
                               np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]))

    #=========================================================================
    # euler sensing
    def test_sensing_euler(self):
        """Is the sensing procedure for Euler equations correct?"""

        # Initialize a mock initial condition that contains a
        # contact discontinuity and a shock
        sol = solution.Solution('entrpyw 12', 'euler', 3)
        gamma = 1.4
        pratio = 20
        Ms = np.sqrt((gamma + 1) / (2.0 * gamma) *
                     pratio + (gamma - 1) / (2.0 * gamma))

        rho0 = 1
        rho1 = 2  # pre-shock density
        u0 = 1
        u1 = u0  # pre-shock velocity
        p0 = 1
        p1 = p0  # pre-shock pressure
        c1 = np.sqrt(gamma * p1 / rho1)

        # post shock
        rho2 = rho1 * (gamma + 1) * Ms * Ms / ((gamma - 1) * Ms * Ms + 2)
        u2 = (1.0 / Ms) * c1 * (2 * (Ms * Ms - 1)) / (gamma + 1)
        p2 = pratio * p1

        # solution all together
        rho = np.array([rho2, rho2, rho2, rho2, rho1, rho1, rho1,
                        rho1, rho1, rho1, rho1, rho0, rho0, rho0])
        u = np.array([u2,  u2,  u2,  u2,  u1,  u1,  u1,
                      u1,  u1,  u1,  u1,  u0,  u0,  u0])
        p = np.array([p2,  p2,  p2,  p2,  p1,  p1,  p1,
                      p1,  p1,  p1,  p1,  p0,  p0,  p0])
        sol.u[:, :] = 0  # initialize the whole solution to zero
        sol.u[0, 0::3] = rho
        sol.u[0, 1::3] = rho * u
        sol.u[0, 2::3] = 0.5 * rho * u * u + p / (gamma - 1)

        # Initialize the sensors
        sen = sensor.Sensor([0.1, 0.4], sol.N_E + 2)

        # Try the sensors
        sen.sensing(sol)

        # test
        npt.assert_array_equal(sen.sensors,
                               np.array([0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 1, 1, 0, 0]))


if __name__ == '__main__':
    unittest.main()
