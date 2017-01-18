#!/usr/bin/env python3
#
#
"""@package dg_stability

Investigate stability of various DG methods and plot the
stability regions.

"""
__author__ = 'Marc T. Henry de Frahan'
__copyright__ = "Copyright (C) 2016, Regents of the University of Michigan"
__license__ = "GPL"
__email__ = "marchdf@umich.edu"
__status__ = "Development"

# ========================================================================
#
# Imports
#
# ========================================================================
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import sympy as smp
from numpy.polynomial import legendre as leg  # import the Legendre functions
from numpy.polynomial import Legendre as L   # import the Legendre class
from numpy import linalg as nla

import aux_functions as auxf

sys.path.insert(0, '../dg1d')
import basis
import enhance

# ========================================================================
#
# Parse arguments
#
# ========================================================================
parser = argparse.ArgumentParser(
    description='A simple plot tool for the one-dimensional DG data')
parser.add_argument('-s', '--show', help='Show the plots', action='store_true')
args = parser.parse_args()


# ========================================================================
#
# Some defaults variables
#
# ========================================================================
plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='Times')
cmap_med = ['#F15A60', '#7AC36A', '#5A9BD4', '#FAA75B',
            '#9E67AB', '#CE7058', '#D77FB4', '#737373']
cmap = ['#EE2E2F', '#008C48', '#185AA9', '#F47D23',
        '#662C91', '#A21D21', '#B43894', '#010202']
dashseq = [(None, None), [10, 5], [10, 4, 3, 4], [
    3, 3], [10, 4, 3, 4, 3, 4], [3, 3], [3, 3]]
markertype = ['s', 'd', 'o', 'p', 'h']

# ========================================================================
#
# Function definitions
#
# ========================================================================


def dg_interior_flux_matrix(p):
    """Calculate the interior flux matrix for the standard DG advection equations

    There is an analytical expression I could use for this too.
    See my thesis page 74.
    """
    F = np.zeros((p + 1, p + 1))
    for i in range(0, p + 1):
        dli = L.basis(i).deriv()
        for j in range(0, p + 1):
            lj = L.basis(j)
            F[i, j] = basis.integrate_legendre_product(dli, lj)

    return F

# ========================================================================


def dg_interface_flux_matrix(p, T):
    """The interface flux matrices, we use upwinding to get the fluxes
     Denote T the translation necessary to evaluate the flux from the
     other cell.

    There are also analytical expressions for these. See my thesis page 73.

    """

    G0 = smp.zeros(p + 1)
    G1 = smp.zeros(p + 1)
    for i in range(0, p + 1):
        for j in range(0, p + 1):
            li = L.basis(i)
            lj = L.basis(j)
            G0[i, j] = leg.legval(-1, li.coef) * \
                leg.legval(1, lj.coef) * (T**-1)
            G1[i, j] = leg.legval(1, li.coef) * leg.legval(1, lj.coef)

    return G0, G1


# ========================================================================
def icb_interface_flux_matrix(p, K, T):
    """The interface flux matrices for the ICB schemes, we use upwinding
     to get the fluxes.  Denote T the translation necessary to
     evaluate the flux from the other cell.

    """

    # Enhanced polynomial degree
    phat = p + len(K)

    G0 = smp.zeros(p + 1, phat + 1)
    G1 = smp.zeros(p + 1, phat + 1)
    for i in range(0, p + 1):
        for j in range(0, phat + 1):
            li = L.basis(i)
            lj = L.basis(j)
            G0[i, j] = leg.legval(-1, li.coef) * \
                leg.legval(1, lj.coef) * (T**-1)
            G1[i, j] = leg.legval(1, li.coef) * leg.legval(1, lj.coef)

    # Enhancement matrix
    A, Ainv, B, Binv = enhance.enhancement_matrices(p, K)

    # Using the enhanced function in the flux (see notes 21/4/15)
    BL = smp.zeros(phat + 1, phat + 1)
    BR = smp.zeros(phat + 1, phat + 1)
    for i in range(p + 1):
        li = L.basis(i)
        BL[i, i] = 1  # basis.integrate_legendre_product(li,li)
        BR[i, i] = 1  # BL[i,i]
    for i, k in enumerate(K):
        lk = L.basis(k)
        int_lklk = basis.integrate_legendre_product(lk, lk)
        BL[i + p + 1, i + p + 1] = T  # * int_lklk
        BR[i + p + 1, i + p + 1] = (T**(-1))  # * int_lklk

    # reduction matrix
    R = smp.zeros(phat + 1, p + 1)
    for i in range(p + 1):
        R[i, i] = 1
    for i, k in enumerate(K):
        for j in range(p + 1):
            R[i + p + 1, j] = auxf.delta(k, j)

    # Convert to sympy matrices
    G0 = smp.Matrix(G0) * smp.Matrix(Ainv) * BL * R
    G1 = smp.Matrix(G1) * smp.Matrix(Ainv) * BL * R

    return G0, G1


# ========================================================================
def dg_matrices(p, T):
    """Get the standard DG discretization matrices"""
    # get the basis functions
    base = basis.Basis(p)

    # Mass matrix and inverse mass matrix
    M, Minv = np.diag(base.m), np.diag(base.minv)

    # Interior flux matrix
    F = dg_interior_flux_matrix(p)

    # Interface flux matrix
    G0, G1 = dg_interface_flux_matrix(p, T)

    return Minv, F, G0, G1


# ========================================================================
def icb_matrices(p, K, T):
    """Get the ICB discretization matrices"""

    # get the basis functions
    base = basis.Basis(p)

    # Mass matrix and inverse mass matrix
    M, Minv = np.diag(base.m), np.diag(base.minv)

    # Interior flux matrix
    F = dg_interior_flux_matrix(p)

    # Interface flux matrix
    G0, G1 = icb_interface_flux_matrix(p, K, T)

    return Minv, F, G0, G1


# ========================================================================
def get_eigenvalue(Minv, F, G0, G1, T):

    N = 50
    betas = np.linspace(-1 * np.pi, np.pi, N)
    ebjs = np.exp(betas * 1j)
    eigs = np.zeros((N, Minv.shape[0]), dtype=np.complex)

    for k, ebj in enumerate(ebjs):

        # replace T symbol by the Fourier transform
        g0 = np.array(G0.subs(T, ebj).tolist()).astype(np.complex)
        g1 = np.array(G1.subs(T, ebj).tolist()).astype(np.complex)

        # Get the full matrix
        FK = Minv.dot(-(g1 - g0) + F)
        eigs[k, :] = nla.eigvals(FK)

    # Sort the eigenvalues into nice arrays
    x, y = auxf.sort_roots_angle(eigs)

    return x, y


# ========================================================================
#
# Basic information/setup
#
# ========================================================================

# Polynomial degree and basis
# orders = [1,2,3,4]
orders = [3]

# Symbols
T = smp.Symbol('T')

# Loop on the orders
for k, p in enumerate(orders):

    # Get the discretization matrices
    # Minv, F, G0, G1 = dg_matrices(p,T)
    Minv, F, G0, G1 = icb_matrices(p, [0, 1, 2], T)

    # Eigenvalues
    x, y = get_eigenvalue(Minv, F, G0, G1, T)
    xy = np.vstack((x, y)).transpose()
    print("For p={0:d},  the stability region intersect at y=0: {1:f}".format(
        p, np.min(x)))
    print("\tThe maximum real part is {0:.12f}".format(np.max(x)))

    # Plot
    plt.figure(0)
    ax = plt.gca()
    polygons = []
    polygons.append(Polygon(xy))
    p = PatchCollection(polygons, edgecolors=cmap[
                        k], linewidths=2, facecolors='none')
    ax.add_collection(p)


# Format the plot
plt.figure(0)
plt.xlabel(r"$\Re(\lambda)$", fontsize=22, fontweight='bold')
plt.ylabel(r"$\Im(\lambda)$", fontsize=22, fontweight='bold')
# plt.xlim([-10,1])
# plt.ylim([-8,8])
plt.axis('equal')
plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight='bold')
plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight='bold')
plt.savefig('dg_stability.pdf', format='pdf')

if args.show:
    plt.show()
