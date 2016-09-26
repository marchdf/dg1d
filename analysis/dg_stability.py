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

#================================================================================
#
# Imports
#
#================================================================================
import argparse
import sys, os
import scipy as sp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.axis as axis
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
#import sympy as sp
#import scipy.optimize
from numpy.polynomial import legendre as leg # import the Legendre functions
from numpy.polynomial import Legendre as L   # import the Legendre class
from numpy import linalg as nla

sys.path.insert(0, '../dg1d')
import basis
import aux_functions as auxf

#================================================================================
#
# Parse arguments
#
#================================================================================
parser = argparse.ArgumentParser(description='A simple plot tool for the one-dimensional DG data')
parser.add_argument('-s','--show', help='Show the plots', action='store_true')
args = parser.parse_args()


#================================================================================
#
# Some defaults variables
#
#================================================================================
plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='Times')
cmap_med =['#F15A60','#7AC36A','#5A9BD4','#FAA75B','#9E67AB','#CE7058','#D77FB4','#737373']
cmap =['#EE2E2F','#008C48','#185AA9','#F47D23','#662C91','#A21D21','#B43894','#010202']
dashseq = [(None,None),[10,5],[10, 4, 3, 4],[3, 3],[10, 4, 3, 4, 3, 4],[3, 3],[3, 3]];
markertype = ['s','d','o','p','h']

#================================================================================
#
# Function definitions
#
#================================================================================
def dg_interior_flux_matrix(p):
    """Calculate the interior flux matrix for the standard DG advection equations
    
    There is an analytical expression I could use for this too. See my thesis page 74.
    """
    F = np.zeros((p+1,p+1))
    for i in range(0,p+1):
        dli = L.basis(i).deriv()
        for j in range(0,p+1):
            lj  = L.basis(j)
            F[i,j] = basis.integrate_legendre_product(dli,lj)

    return F
    
#================================================================================
def dg_interface_flux_matrix(p):
    """The interface flux matrices, we use upwinding to get the fluxes
     Denote T the translation necessary to evaluate the flux from the
     other cell.

    There are also analytical expressions for these. See my thesis page 73.

    """

    G0 = np.zeros((p+1,p+1))
    G1 = np.zeros((p+1,p+1))
    for i in range(0,p+1):
        for j in range(0,p+1):
            li  = L.basis(i)
            lj  = L.basis(j)
            G0[i,j] = leg.legval(-1,li.coef) * leg.legval(1,lj.coef) #*(T**-1)
            G1[i,j] = leg.legval( 1,li.coef) * leg.legval(1,lj.coef)

    return G0,G1


#================================================================================
def get_eigenvalue(Minv,F,G0,G1):

    N = 50
    betas = np.linspace(-1*np.pi,np.pi,N)
    eigs = np.zeros((N,Minv.shape[0]),dtype=np.complex)
    
    for k,beta in enumerate(betas):
        ebj = np.exp(beta*1j)
        FK = Minv.dot(-(G1 - G0/ebj) + F)
        eigs[k,:] = nla.eigvals(FK)

    # Sort the eigenvalues into nice arrays
    x, y = auxf.sort_roots_angle(eigs)
        
    return x,y
        

#================================================================================
#
# Basic information/setup
#
#================================================================================

# Polynomial degree and basis
orders = [1,2,3,4]

# Loop on the orders
for k,p in enumerate(orders):

    # get the basis functions
    base = basis.Basis(p)

    # Mass matrix and inverse mass matrix
    M, Minv = np.diag(base.m), np.diag(base.minv)

    # Interior flux matrix
    F = dg_interior_flux_matrix(p)

    # Interface flux matrix
    G0,G1 = dg_interface_flux_matrix(p)

    # Eigenvalues
    x,y = get_eigenvalue(Minv,F,G0,G1)
    xy = np.vstack((x,y)).transpose()
    print("For p={0:d}, estimate of where the stability region intersects y=0 axis: {1:f}".format(p,np.min(x)))

    # Plot
    plt.figure(0)
    ax = plt.gca()
    polygons = [];
    polygons.append(Polygon(xy))
    p = PatchCollection(polygons, edgecolors = cmap[k], linewidths = 2, facecolors = 'none')
    ax.add_collection(p)


# Format the plot
plt.figure(0)
plt.axis('equal')
plt.xlabel(r"$\Re(\lambda)$",fontsize=22,fontweight='bold')
plt.ylabel(r"$\Im(\lambda)$",fontsize=22,fontweight='bold')
plt.setp(ax.get_xmajorticklabels(),fontsize=18,fontweight='bold');
plt.setp(ax.get_ymajorticklabels(),fontsize=18,fontweight='bold');
plt.savefig('dg_stability.pdf',format='pdf')

if args.show:
    plt.show()
