#!/usr/bin/env python
#
#
"""@package rk_stability

Investigate stability of various (explicit) RK methods and plot the
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
import sympy as sp
import scipy.optimize


sys.path.insert(0, '../dg1d')
import rk_coeffs as rkc


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
def sort_roots_angle(roots):
    """Radial sort of the roots

    Inspired from:
    http://stackoverflow.com/questions/35606712/numpy-way-to-sort-out-a-messy-array-for-plotting

    """
    # Get one long array with all the roots
    reshaped = roots.reshape((roots.size,),order='F')
    
    # Separate the real and imaginary parts
    x, y = np.real(reshaped), np.imag(reshaped)

    # Get the angle wrt the mean of the cloud of points
    x0, y0 = x.mean(), y.mean()
    angle = np.arctan2(y - y0, x - x0)

    # Sort based on this angle
    idx = angle.argsort()

    return x[idx], y[idx]

#================================================================================
def exact_stability_boundary(coeffs,B,fignum=0,coloridx=0):
    """Returns points on the stability boundary of an explicit RK scheme

    coeffs is the bottom row of the Butcher table
    B is the matrix in the Butcher table
    
    NB: this will fail for a high-order RK scheme since we are taking
    a determinant of a symbolic matrix.

    """

    # Number of stages
    s = len(coeffs)

    # Vector of ones and identity matrix
    e = np.ones(s)
    I = np.diag(e)

    # Construct the polynomial expression (stability function)
    z = sp.Symbol('z')
    A = sp.Matrix(I - z * B + z * np.outer(e,coeffs))
    sigma = sp.Poly(sp.det(A))

    # Discretize the angles and initialize storage
    nroots = sp.degree(sigma)
    N = 100
    thetas = np.linspace(0,2*np.pi,N)
    roots  = np.zeros((N,nroots),dtype=np.complex_)
    
    # Loop on all the angles
    for k, theta in enumerate(thetas):
        
        # Get the polynomial expression with the zeroth-degree adjusted by
        # the angle
        pp = np.array(sigma.coeffs(),dtype=np.complex_)
        pp[-1] += np.exp(1j*theta)
        
        # Get the roots
        roots[k,:] = np.roots(np.array(pp))
        
        
    # Sort the roots
    x,y = sort_roots_angle(roots)  

    # Plot
    plt.figure(fignum)
    plt.fill(x,y,ec=cmap[coloridx],fill=False,lw=2)
    # for n in range(nroots):
    #     plt.plot(np.real(roots[:,n]),np.imag(roots[:,n]),'o',mec=cmap[n],mfc=cmap[n],ms=5)

    return x,y




#================================================================================
def approximate_stability_boundary(coeffs,B,fignum=0,coloridx=0,pltlabel=''):
    """Finds the approximate stability boundary of an explicit RK scheme

    coeffs is the bottom row of the Butcher table
    B is the matrix in the Butcher table

    """

    # Number of stages
    s = len(coeffs)

    # Vector of ones and identity matrix
    e = np.ones(s)
    I = np.diag(e)

    # Construct the polynomial expression (stability function)
    num = 100
    zrs = np.linspace(-3.5,1,num)
    zis = np.linspace(0,3.5,num)
    ZR, ZI = np.meshgrid(zrs,zis)
    Z = ZR + 1j * ZI
    sigma = np.zeros((num,num))
    
    for i in range(num):
        for j in range(num):
            A = I - Z[i,j] * B + Z[i,j] * np.outer(e,coeffs)
            sigma[i,j] = abs(np.linalg.det(A))

    # Plot the contour
    plt.figure(fignum)
    cs1 = plt.contour(ZR,ZI,sigma,[1],colors=cmap[coloridx],linewidths=2)
    cs2 = plt.contour(ZR,-ZI,sigma,[1],colors=cmap[coloridx],linewidths=2)
    #plt.clabel(cs1,cs1.levels,inline=True,fmt = pltlabel, fontsize=10)
    p = cs1.collections[0].get_paths()[0]
    v = p.vertices
    x = v[:,0]
    y = v[:,1]
           
    return x,y, cs1.collections[0]


#================================================================================
def get_formatted_rk_coeffs(method='rk4'):
    """Returns the RK coefficients formatted so that we can use them"""

    # Get the coefficients
    if method == 'rk3':
        coeffs, alphas, betas = rkc.get_rk3_coefficients()
    elif method == 'rk4':
        coeffs, alphas, betas = rkc.get_rk4_coefficients()
    elif method == 'rk10':
        coeffs, alphas, betas = rkc.get_rk10_coefficients()
    elif method == 'rk12':
        coeffs, alphas, betas = rkc.get_rk12_coefficients()
    elif method == 'rk14':
        coeffs, alphas, betas = rkc.get_rk14_coefficients()
    else:
        print('Wrong RK method, defaulting to RK4')
        coeffs, alphas, betas = rkc.get_rk4_coefficients()
        
    coeffs = np.array(coeffs)
    
    # Number of stages
    s = len(coeffs)
    
    # Get the B matrix (formed of the betas)
    k = np.array([np.zeros(s)])
    B = np.concatenate((betas,k.T),axis=1)

    return coeffs, B

    
#================================================================================
#
# Problem setup
#
#================================================================================

# Plot the exact stability boundary for RK4
#coeffs, B = get_formatted_rk_coeffs()
#x,y = exact_stability_boundary(coeffs,B,0,0)


# Methods to plot
methods = ['rk3','rk4','rk10','rk12','rk14']
ps = [None] * len(methods)

# Loop over these methods
for k,method in enumerate(methods):
    coeffs, B = get_formatted_rk_coeffs(method)
    x,y,ps[k] = approximate_stability_boundary(coeffs,B,0,k,method);

# Format figure
plt.figure(0)
ax = plt.gca()
plt.legend(ps, [m.upper() for m in methods])
plt.axis('equal')
plt.xlabel(r"$\Re(\lambda \Delta t)$",fontsize=22,fontweight='bold')
plt.ylabel(r"$\Im(\lambda \Delta t)$",fontsize=22,fontweight='bold')
plt.setp(ax.get_xmajorticklabels(),fontsize=18,fontweight='bold');
plt.setp(ax.get_ymajorticklabels(),fontsize=18,fontweight='bold');
plt.savefig('rk_stability.pdf',format='pdf')

if args.show:
    plt.show()

