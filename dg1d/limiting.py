# ========================================================================
#
# Imports
#
# ========================================================================
import sys
import numpy as np
from numpy.polynomial import legendre as leg  # import the Legendre functions

import dg1d.solution as solution

# ========================================================================
#
# Class definitions
#
# ========================================================================


class Limiter:
    'Generate the limiter'

    # ========================================================================
    def __init__(self, limiting_type, solution):

        print("Setting up the limiter:")

        # Pre-allocate depending on limiting type
        self.keywords = {'type': None}

        if limiting_type == 'adaptive_hr':
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
                        m - 1, n] = integrate_monomial_derivative(m - 1, n)
                    if(n >= m + 1):
                        self.integral_monomial_derivative_bounds_31[
                            m - 1, n] = integrate_monomial_derivative_bounds(m - 1, n, -3, -1)
                        self.integral_monomial_derivative_bounds_13[
                            m - 1, n] = integrate_monomial_derivative_bounds(m - 1, n, 1, 3)

        # By default, do not limit
        else:
            print('\tNo limiting.')

    # ========================================================================
    def limit(self, solution):
        """Limit a solution"""

        if self.keywords['type'] is not None:
            self.keywords['type'](solution)

    # ========================================================================
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

    # ========================================================================
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

    # ========================================================================
    def legendre_to_monomial(self, l):
        """Transform a Legendre solution to a monomial (Taylor series) representation"""
        return np.dot(self.L2M, l)

    # ========================================================================
    def monomial_to_legendre(self, t):
        """Transform a monomial (Taylor series) solution to a Legendre representation"""
        return np.dot(self.M2L, t)

    # ========================================================================
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
            alim[m] = scalar_minmod(c1, c2)

        # preserve cell average
        alim[0] = avgLC
        return alim


# ========================================================================


def integrate_monomial_derivative(k, n):
    r"""The integral of the kth derivative of nth order monomial (from -1 to 1)

    Returns :math:`\frac{2}{(n-k+1)!}` if :math:`n-k+1` is odd, 0 otherwise

    Basically, calculates :math:`\int_{-1}^1 \frac{\partial^k}{\partial x^k} \frac{x^n}{n!} \mathrm{d} x`
    """
    num = n - k + 1
    if (num % 2):
        return 2.0 / np.math.factorial(num)
    else:
        return 0.0

# ========================================================================


def integrate_monomial_derivative_bounds(k, n, a, b):
    r"""The integral of the kth derivative of nth order monomial (from a to b)

    Returns :math:`\int_{a}^{b} \frac{\partial^k}{\partial x^k} \frac{x^n}{n!} \mathrm{d} x`
        """
    num = n - k + 1
    return (b**num - a**num) / np.math.factorial(num)

# ========================================================================


def scalar_minmod(a, b):
    """Minmod function for two scalars

    Idea from http://codegolf.stackexchange.com/questions/42079/shortest-minmod-function
    """
    return sorted([a, b, 0])[1]
