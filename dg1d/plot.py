#!/usr/bin/env python
#
#
"""@package sample plotting

A sample plotting tool for the one-dimensional DG data. The exact
solution plotted is for the sinewave initial condition.

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

import solution
import dg

#================================================================================
#
# Parse arguments
#
#================================================================================
parser = argparse.ArgumentParser(description='A simple plot tool for the one-dimensional DG data')
parser.add_argument('-s','--show', help='Show the plots', action='store_true')
parser.add_argument('-f','--file', dest='fname', help='File to load', metavar = "FILE", required=True)
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
# Problem setup
#
#================================================================================

solution = solution.Solution('empty','empty',0)
solution.loader(args.fname)

# Collocate the solution to the Gaussian nodes
ug = solution.collocate()

# Collocate to the cell edge values
uf = solution.evaluate_faces()

# Plot each element solution in a different color
for e in range(solution.N_E):
    a = solution.x[e]
    b = solution.x[e+1]
    xg = 0.5*(b-a)*solution.basis.x + 0.5*(b+a)

    # plot the solution at the Gaussian nodes (circles)
    plt.plot(xg,ug[:,e],'o',mfc=cmap[e%len(cmap)],mec=cmap[e%len(cmap)])

    # Plot the solution at the cell edges (squares)
    plt.plot([a,b],uf[:,e],'s',mfc=cmap[e%len(cmap)],mec='black')

# Plot the exact solution
xe = np.linspace(-1,1,200)
fe = np.sin(2*np.pi*xe)
plt.plot(xe,fe,'k')

if args.show:
    plt.show()
