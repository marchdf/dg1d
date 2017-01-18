#=========================================================================
#
# Imports
#
#=========================================================================
import sys
import numpy as np
from numpy.polynomial import legendre as leg  # import the Legendre functions

import dg1d.solution as solution

#=========================================================================
#
# Class definitions
#
#=========================================================================


class Limiter:
    'Generate the limiter'

    #=========================================================================
    def __init__(self, limiting_type, solution):

        print("Setting up the limiter:")

        # Pre-allocate depending on limiting type
        self.keywords = {'type': None}

        if limiting_type == 'full_biswas':
            print('\tLimiting everywhere with Biswas limiter')
            self.keywords = {'type': self.full_biswas}

            # Pre-allocate vector holding 1/(2*k+1). The little None
            # business is to do the multiplication broadcasting with the
            # delta. According to
            # http://stackoverflow.com/questions/16229823/how-to-multiply-numpy-2d-array-with-numpy-1d-array
            self.c = 1.0 / (2 * np.arange(0, solution.basis.p) + 1)[:, None]

        elif limiting_type == 'adaptive_hr':
            print('\tAdaptive limiting with hierarchical reconstruction')
            self.keywords = {'type': self.adaptive_hr}

            self.ulim = np.zeros(solution.u.shape)

            # Pre-allocate basis transforms. We don't use the builtin
            # python ones because they are slow!
            V = np.zeros((solution.basis.N_s, solution.basis.N_s))
            x = solution.basis.x
            for i in range(solution.basis.N_s):
                for j in range(solution.basis.N_s):
                    V[i, j] = x[i]**j / np.math.factorial(j)

            self.L2M = np.dot(np.linalg.inv(V), solution.basis.phi)
            self.M2L = np.linalg.inv(self.L2M)

            # Pre-allocate some common integrals we need to limit
            self.integral_monomial_derivative = np.zeros(
                (solution.basis.N_s, solution.basis.N_s))
            self.integral_monomial_derivative_bounds_31 = np.zeros(
                (solution.basis.N_s, solution.basis.N_s))
            self.integral_monomial_derivative_bounds_13 = np.zeros(
                (solution.basis.N_s, solution.basis.N_s))
            for m in range(solution.basis.p, 0, -1):
                for n in range(m - 1, solution.basis.p + 1):
                    self.integral_monomial_derivative[
                        m - 1, n] = self.integrate_monomial_derivative(m - 1, n)
                    if(n >= m + 1):
                        self.integral_monomial_derivative_bounds_31[
                            m - 1, n] = self.integrate_monomial_derivative_bounds(m - 1, n, -3, -1)
                        self.integral_monomial_derivative_bounds_13[
                            m - 1, n] = self.integrate_monomial_derivative_bounds(m - 1, n, 1, 3)

        # By default, do not limit
        else:
            print('\tNo limiting.')

    #=========================================================================
    def limit(self, solution):
        """Limit a solution"""

        if self.keywords['type'] is not None:
            self.keywords['type'](solution)

    #=========================================================================
    def full_biswas(self, solution):
        """Limit a solution everywhere in the domain using the Biswas limiter"""

        # Get the differences with the left/right neighbors
        deltam, deltap = self.deltas(solution)

        # Apply the minmod procedure (this is incorrect as it applies it to all the modes)
        #solution.u[1:,solution.N_F:-solution.N_F] = self.minmod(deltam,deltap,solution.u[1:,solution.N_F:-solution.N_F])

        # This is the correct Biswas limiter as it stops limiting
        # when it is no longer necessary. But this is a very ugly
        # implementation
        result = self.minmod(deltam, deltap, solution.u[
                             1:, solution.N_F:-solution.N_F])
        for e in range(1, solution.N_E + 1):
            for f in range(0, solution.N_F):

                stop = False
                for k in range(solution.basis.p, 0, -1):
                    if (np.fabs(solution.u[k, e * solution.N_F + f] - result[k - 1, (e - 1) * solution.N_F + f]) < 1e-14) or stop:
                        stop = True
                    else:
                        solution.u[k, e * solution.N_F +
                                   f] = result[k - 1, (e - 1) * solution.N_F + f]

    #=========================================================================
    def deltas(self, solution):
        """Calculate the difference between left and right neighbors"""

        # Total number of elements in the solution (including ghosts)
        total_num_element = solution.u.shape[1]

        # Index of all elements to be limited
        idx = np.arange(solution.N_F, total_num_element - solution.N_F)

        # Index of their neighbors to the left
        idxl = idx - solution.N_F

        # Index of their neighbors to the right
        idxr = idx + solution.N_F

        # Differences with the left and right neighbors
        deltam = (solution.u[:-1, idx] - solution.u[:-1, idxl]) * self.c
        deltap = (solution.u[:-1, idxr] - solution.u[:-1, idx]) * self.c

        return deltam, deltap

    #=========================================================================
    def minmod(self, A, B, C):
        """Given three arrays do an element-by-element minmod

        For each entry a, b, c in A, B,C, return:
           max(a,b,c) if a,b,c < 0
           min(a,b,c) if a,b,c > 0
           0          otherwise

        There is another way of doing this. Here, we are getting the
        max and min everywhere in the array, then deciding where to
        used the min/max (depending on the sign of the
        elements). Another way would be to find the indices where the
        min/max should be taken, then take the min/max for those
        columns. I ran some tests and (surprisingly?) the faster way
        is actually to do it the first way (min/max then decide where
        to use them). I left the other way below commented.
        """

        # Initialize
        M = np.zeros(A.size)
        D = np.vstack((A.flatten(), B.flatten(), C.flatten()))

        # Find where indices where they are all positive or all
        # negative and we should be getting the max or min
        idxmax = np.where(np.all(D < 0, axis=0))
        idxmin = np.where(np.all(D > 0, axis=0))

        # Find the max and min comparing a,b,c
        maxi = np.max(D, axis=0)
        mini = np.min(D, axis=0)

        # Populate the minmod matrix with the results
        M[idxmax] = maxi[idxmax]
        M[idxmin] = mini[idxmin]

        # I initially thought this might be faster. But it's not. I am
        # leaving this here because this result is a bit surprising to
        # me. In theory, you are doing fewer min/max operations
        # compared to the other way.
        # # Find the max and min comparing a,b,c
        # maxi = np.max(D[:,idxmax],axis=0)
        # mini = np.min(D[:,idxmin],axis=0)
        # # Populate the minmod matrix with the results
        # M[idxmax] = maxi
        # M[idxmin] = mini

        # Reshape the matrix and return the results
        return M.reshape(A.shape)

    #=========================================================================
    def adaptive_hr(self, solution):
        """Limit a solution in the domain using adaptive hierarchical reconstruction"""

        # Decide where to do limiting
        solution.sensors.sensing(solution)

        # loop over all the interior elements
        self.ulim = np.copy(solution.u)
        for e in range(1, solution.N_E + 1):
            # for e in range(solution.N_E,0,-1):

            # test if we need to do limiting (sensors are on)
            if solution.sensors.sensors[e] != 0:

                # loop over the fields and call HR
                for f in range(0, solution.N_F):
                    self.ulim[:, e * solution.N_F + f] = self.hr(solution.u[:, e * solution.N_F + f],
                                                                 solution.u[
                                                                     :, (e - 1) * solution.N_F + f],
                                                                 solution.u[:, (e + 1) * solution.N_F + f])

        solution.u = np.copy(self.ulim)
        solution.apply_bc()

    #=========================================================================
    def hr(self, uc, ul, ur):
        """Limit a cell solution with hierarchical reconstruction"""

        # Legendre -> monomial transform
        uc = self.legendre_to_monomial(uc)
        ul = self.legendre_to_monomial(ul)
        ur = self.legendre_to_monomial(ur)

        # Limit a monomial solution
        uc_lim = self.limit_monomial(uc, ul, ur)

        # monomial -> Legendre transform
        uc_lim = self.monomial_to_legendre(uc_lim)

        return uc_lim

    #=========================================================================
    def legendre_to_monomial(self, l):
        """Transform a Legendre solution to a monomial (Taylor series) representation"""
        return np.dot(self.L2M, l)

    #=========================================================================
    def monomial_to_legendre(self, t):
        """Transform a monomial (Taylor series) solution to a Legendre representation"""
        return np.dot(self.M2L, t)

    #=========================================================================
    def limit_monomial(self, ac, al, ar):
        """Limit a monomial cell solution with hierarchical reconstruction"""

        alim = np.zeros(np.shape(ac))

        # loop on derivatives
        N = len(ac) - 1
        for m in range(N, 0, -1):

            # Initialize
            avgdUL = 0
            avgdUC = 0
            avgdUR = 0
            avgRL = 0
            avgRC = 0
            avgRR = 0

            # Calculate the derivative average in the cells: left,
            # center, right. Calculate the remainder polynomial in our
            # cells and its two neighbors
            for n in range(m - 1, N + 1):
                integral = self.integral_monomial_derivative[m - 1, n]
                avgdUL += al[n] * integral
                avgdUC += ac[n] * integral
                avgdUR += ar[n] * integral
                if(n >= m + 1):
                    avgRL += alim[n] * \
                        self.integral_monomial_derivative_bounds_31[m - 1, n]
                    avgRC += alim[n] * integral
                    avgRR += alim[n] * \
                        self.integral_monomial_derivative_bounds_13[m - 1, n]

            # Approximate the average of the linear part
            # avg = \frac{1}{2} \int_{-1}^1 U \ud x
            avgLL = 0.5 * (avgdUL - avgRL)
            avgLC = 0.5 * (avgdUC - avgRC)
            avgLR = 0.5 * (avgdUR - avgRR)

            # MUSCL approach to get candidate coefficients
            c1 = 0.5 * (avgLC - avgLL)  # 1/dx = 1/2 = 0.5
            c2 = 0.5 * (avgLR - avgLC)

            # Limited value
            alim[m] = self.scalar_minmod(c1, c2)

        # preserve cell average
        alim[0] = avgLC
        return alim

    #=========================================================================
    def integrate_monomial_derivative(self, k, n):
        r"""The integral of the kth derivative of nth order monomial (from -1 to 1)

        Returns :math:`\frac{2}{(n-k+1)!}` if :math:`n-k+1` is odd, 0 otherwise

        Basically, calculates :math:`\int_{-1}^1 \frac{\partial^k}{\partial x^k} \frac{x^n}{n!} \mathrm{d} x`
        """
        num = n - k + 1
        if (num % 2):
            return 2.0 / np.math.factorial(num)
        else:
            return 0.0

    #=========================================================================
    def integrate_monomial_derivative_bounds(self, k, n, a, b):
        r"""The integral of the kth derivative of nth order monomial (from a to b)

        Returns :math:`\int_{a}^{b} \frac{\partial^k}{\partial x^k} \frac{x^n}{n!} \mathrm{d} x`
        """
        num = n - k + 1
        return (b**num - a**num) / np.math.factorial(num)

    #=========================================================================
    def scalar_minmod(self, a, b):
        """Minmod function for two scalars

        Idead from http://codegolf.stackexchange.com/questions/42079/shortest-minmod-function
        """
        return sorted([a, b, 0])[1]
