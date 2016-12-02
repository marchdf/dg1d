#================================================================================
#
# Imports
#
#================================================================================
import unittest
import euler_physics
import numpy as np
import numpy.testing as npt

#================================================================================
#
# Class definitions
#
#================================================================================
class EulerPhysicsTestCase(unittest.TestCase):
        """Tests for `euler_physics.py`."""

        #================================================================================
        # max_wave_speed
        def test_max_wave_speed(self):
                """Is the max wave speed solver correct?"""

                # toy data
                u = np.arange(1,12*3+1).reshape((3,12))

                # Get the maximum wave speed
                m = euler_physics.max_wave_speed(u)

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
                F = euler_physics.riemann_rusanov(ul,ur)

                # test
                npt.assert_array_almost_equal(F,
                                              np.array([ -8.61582313, -4.13415831, -0.72679906, 1.78892781, 5.11780113, 7.24999235, 7.3690381, 10.53092381, 12.48640363, 12.37032176, 15.52088988,  17.51156911]),
                                              decimal = 7)
            
        #================================================================================
        # riemann_godunov
        def test_riemann_godunov(self):
                """Is the Godunov Riemann solver correct?"""

                # Left/right toy data taken from shock tube problems found here:
                # http://num3sis.inria.fr/blog/eulerian-flows-approximate-riemann-solvers-validation-on-1d-test-cases/
                # or: C. Kong MS thesis at U. of Reading http://www.readingconnect.net/web/FILES/maths/CKong-riemann.pdf

                # For each test: rhoL, uL, pL, rhoR, uR, pR
                t1 = np.array([      1,         0,       1,   0.125,     0.0,     0.1])    # Sod shock tube
                t2 = np.array([      1,      0.75,     1.0,   0.125,     0.0,     0.1])    # Modified Sod shock tube
                t3 = np.array([      1,      -2.0,     0.4,     1.0,     2.0,     0.4])    # 123 problem
                t4 = np.array([      1,       0.0,  1000.0,     1.0,     0.0,    0.01])    # Left Woodward and Colella (blast wave)
                t5 = np.array([5.99924,   19.5975, 460.894, 5.99242,-6.19633, 46.0950])    # collision of two strong shocks
                t6 = np.array([    1.4,       0.0,     1.0,     1.0,     0.0,     1.0])    # stationary contact discontinuity

                # Transform to conserved variables
                def ptoc(t):
                        """Take a test case containing rhoL, uL, pL, rhoR, uR, pR and turn them into conservative variables"""
                        gamma = 1.4
                        t[1] = t[0]*t[1]
                        t[2] = t[2]/(gamma-1) + 0.5*t[0]*t[1]*t[1]
                        t[4] = t[3]*t[4]
                        t[5] = t[5]/(gamma-1) + 0.5*t[3]*t[4]*t[4]
                        return t

                t1 = ptoc(t1)
                t2 = ptoc(t2)
                t3 = ptoc(t3)
                t4 = ptoc(t4)
                t5 = ptoc(t5)
                t6 = ptoc(t6)
                
                # Put them into a long vector
                ul = np.array([t1[0:3],
                               t2[0:3],
                               t3[0:3],
                               t4[0:3],
                               t5[0:3],
                               t6[0:3]]).flatten()
                ur = np.array([t1[3::],
                               t2[3::],
                               t3[3::],
                               t4[3::],
                               t5[3::],
                               t6[3::]]).flatten()
            
                # Get the flux
                F = euler_physics.riemann_godunov(ul,ur)
                print(F)

                print(np.array([0.3953910704650308,0.6698366621333465,1.1540375166808616,
                        0.8109525650238815,1.5445355710738495,3.0029992255123030,
                                0.0000000000000000,0.0018938734200542,0.0000000000000000,
                                11.2697554398918438,681.7522718876612089,33777.3342909460770898,
                                117.5701059000000015,2764.9741503752502467,54190.4009509894967778,
                                0.0000000000000000,1.0000000000000000,0.0000000000000000]))
                
                # test (exact solution generated by my exact Riemann solver)
                npt.assert_array_almost_equal(F,
                                              np.array([0.3953910704650308,0.6698366621333465,1.1540375166808616,
                                                        0.8109525650238815,1.5445355710738495,3.0029992255123030,
                                                        0.0000000000000000,0.0018938734200542,0.0000000000000000,
                                                        11.2697554398918438,681.7522718876612089,33777.3342909460770898,
                                                        117.5701059000000015,2764.9741503752502467,54190.4009509894967778,
                                                        0.0000000000000000,1.0000000000000000,0.0000000000000000]),
                                              decimal = 7)


        #================================================================================
        # riemann_roe
        def test_riemann_roe(self):
                """Is the Roe Riemann solver correct?"""

                # Left/right toy data
                ul = np.arange(1,13)
                ur = np.arange(1,13)[::-1]
            
                # Get the flux
                F = euler_physics.riemann_roe(ul,ur)

                # test
                npt.assert_array_almost_equal(F,
                                              np.array([  2., 4.4, 6.8, 5., 7.4, 8.9375, 8., 10.9142857, 12.3102041, 11., 14.48, 15.818]),
                                              decimal = 7)


        #================================================================================
        # interior_flux
        def test_interior_flux(self):
                """Is the interior flux correct?"""

                # toy data
                u = np.arange(1,12*3+1).reshape((3,12))

                # Get the maximum wave speed
                F = euler_physics.interior_flux(u)
            
                # test
                npt.assert_array_almost_equal(F,
                                              np.array([[ 2., 4.4, 6.8, 5., 7.4, 8.9375, 8., 10.91428571, 12.31020408, 11., 14.48, 15.818],
                                                        [ 14., 18.06153846, 19.36804734, 17., 21.65, 22.93671875, 20., 25.24210526, 26.51523546, 23., 28.83636364, 30.09958678],
                                                        [ 26., 32.432, 33.68768, 29., 36.02857143, 37.27831633, 32., 39.62580645, 40.87075963, 35., 43.22352941, 44.46453287]]),
                                              decimal = 7)
            

if __name__ == '__main__':
        unittest.main()

